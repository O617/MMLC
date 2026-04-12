# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain VLM (ViT+MLP+LLM) MODEL."""
from copy import deepcopy
from functools import partial
from typing import Dict, Any

from datasets import Dataset
import torch
import os

import mindspeed.megatron_adaptor
from mindspeed.megatron_adaptor import get_mindspeed_args
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args, print_rank_0
from megatron.training.utils import average_losses_across_data_parallel_group
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.vlm_model import VLMModel
from mindspeed_mm.patchs import dummy_optimizer_patch
from mindspeed_mm.training import pretrain
from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.utils.hetero_parallel import change_parallel_state, apply_hetero_parallel_hooks
from mindspeed_mm.utils.utils import EncoderBalanceComm
from mindspeed_mm.utils.hetero_parallel import hetero_align_config
from mindspeed_mm.utils.utils import compute_token_level_loss
from scheduler import Scheduler, DoubleBufferedScheduler
from profiler import MLLMProfiler
mindspeed_args = get_mindspeed_args()
data_scheduler = None
hybrid_parallel = os.environ.get("HYBRID_PARALLEL")
# Double-buffered async prefetch state
_prefetch_raw_batch = None  # raw batch pre-fetched for the next step
_prefetch_started = False   # whether start_prefetch has been called
if hasattr(mindspeed_args, "ai_framework") and mindspeed_args.ai_framework == "mindspore" and mindspeed_args.optimization_level >= 0:
    import mindspeed_mm.mindspore.mindspore_adaptor


def model_provider(pre_process=True, post_process=True, modules=None):
    """Builds the model."""
    if modules is None:
        modules = ['image_encoder', 'audio_encoder', 'text_decoder']

    args = get_args()
    print_rank_0("building VLMModel ...")
    vlm_config = deepcopy(args.mm.model)

    # distinguish model construct stage when pipeline parallel
    vlm_config.pre_process = pre_process
    vlm_config.post_process = post_process

    _configure_modules(vlm_config, modules)

    model = VLMModel(vlm_config)
    # MLLMProfiler("InternVL", model, get_args(),)

    if args.hetero_parallel:
        print_rank_0("apply hetero parallel ...")
        apply_hetero_parallel_hooks(model)

    _apply_freezing(model, vlm_config)

    global data_scheduler
    other_parallel_group_size = mpu.get_tensor_model_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size()
    use_async = os.environ.get("ASYNC_SCHEDULE", "False") == "True"
    scheduler_cls = DoubleBufferedScheduler if use_async else Scheduler
    scheduler_kwargs = dict(
        cluster_size=torch.distributed.get_world_size(),
        other_parallel_group_size=other_parallel_group_size,
        img_context_token_id=model.img_context_token_id,
        cp_window_size=get_args().context_parallel_size,
        schedule_mode=os.environ.get("SCHEDULE_MODE", "dynamic"),
    )
    data_scheduler = scheduler_cls(**scheduler_kwargs)

    if hybrid_parallel is not None and hybrid_parallel == "True":
        from megatron.core.num_microbatches_calculator import reconfigure_num_microbatches_calculator
        other_ps = mpu.get_tensor_model_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size()
        num_groups = torch.distributed.get_world_size() // other_ps // args.context_parallel_size
        reconfigure_num_microbatches_calculator(
            rank=torch.distributed.get_rank(),
            rampup_batch_size=None,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            data_parallel_size=num_groups,
        )

    return model


def _configure_modules(vlm_config, modules):
    """Configure each module based on the modules list."""
    module_configs = {
        'image_encoder': _configure_image_encoder,
        'audio_encoder': _configure_audio_encoder,
        'text_decoder': _configure_text_decoder
    }

    for module_name, config_func in module_configs.items():
        if module_name in modules and hasattr(vlm_config, module_name):
            config_func(vlm_config)
        else:
            setattr(vlm_config, module_name, None)


def _configure_image_encoder(vlm_config):
    """Configure image encoder module."""
    if get_args().hetero_parallel:
        hetero_align_config(vlm_config.image_encoder.vision_encoder, vlm_config.image_encoder)
        hetero_align_config(vlm_config.image_encoder.vision_projector, vlm_config.image_encoder)

    # MindSpeed needs to validate the CP configuration; the attention head must be divisible by the CP sizes.
    # However, since the vision projector does not have an attention head, special handling is required.
    vlm_config.image_encoder.vision_projector.context_parallel_size = 1
    vlm_config.image_encoder.vision_encoder.expert_model_parallel_size = 1
    vlm_config.image_encoder.vision_projector.expert_model_parallel_size = 1
    vlm_config.image_encoder.vision_encoder = get_model_config(vlm_config.image_encoder.vision_encoder)
    vlm_config.image_encoder.vision_projector = get_model_config(vlm_config.image_encoder.vision_projector)


def _configure_audio_encoder(vlm_config):
    """Configure audio encoder module."""
    if get_args().hetero_parallel:
        hetero_align_config(vlm_config.audio_encoder.audio_encoder, vlm_config.audio_encoder)

    vlm_config.audio_encoder.audio_encoder = get_model_config(vlm_config.audio_encoder.audio_encoder)


def _configure_text_decoder(vlm_config):
    """Configure text decoder module."""
    if get_args().hetero_parallel:
        hetero_align_config(vlm_config.text_decoder, vlm_config.text_decoder)
        
    vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)


def _apply_freezing(model, vlm_config):
    """Apply freezing settings to the model."""
    has_image = hasattr(vlm_config, 'image_encoder') and vlm_config.image_encoder is not None
    freeze_image_encoder = has_image and getattr(vlm_config.image_encoder.vision_encoder, 'freeze', True)
    freeze_image_projection = has_image and getattr(vlm_config.image_encoder.vision_projector, 'freeze', False)

    has_audio = hasattr(vlm_config, 'audio_encoder') and vlm_config.audio_encoder is not None
    freeze_audio_encoder = has_audio and getattr(vlm_config.audio_encoder.audio_encoder, 'freeze', True)

    model.freeze(
        freeze_image_encoder=freeze_image_encoder,
        freeze_image_projection=freeze_image_projection,
        freeze_audio_encoder=freeze_audio_encoder
    )


def move_to_device(batch: Dict[str, Any], float_dtype: str):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            batch[k] = v.to(device=torch.cuda.current_device(), dtype=dtype)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            batch[k] = [t.to(device=torch.cuda.current_device(),
                             dtype=float_dtype if torch.is_floating_point(t) else None)
                        for t in v]


def _load_raw_batch(data_iterator):
    """Load a raw batch from the data iterator and move to device."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    move_to_device(batch, get_args().params_dtype)
    has_video = 'pixel_values_videos' in batch and 'video_grid_thw' in batch
    if has_video:
        batch['pixel_values'] = batch.pop('pixel_values_videos')
        batch['image_grid_thw'] = batch.pop('video_grid_thw')
    return batch


def _apply_encoder_balance(batch, is_vit_last_stage):
    """Apply encoder DP balance communication if needed."""
    if (mpu.is_pipeline_first_stage() or is_vit_last_stage) and get_args().encoder_dp_balance:
        batch['pixel_values'], batch['tranfer'] = EncoderBalanceComm.apply(
            batch['pixel_values'],
            mpu.get_data_parallel_group())
    else:
        batch['tranfer'] = None


def _diag_raw_batch(batch):
    """Print diagnostic info for raw batch (before Scheduler)."""
    _rank = torch.distributed.get_rank()
    _raw_labels = batch.get('labels', None)
    if _raw_labels is not None:
        _raw_shape = tuple(_raw_labels.shape)
        _raw_valid = int((_raw_labels > -1).sum())
        _flat = _raw_labels.flatten()
        _nonpad = _flat[_flat > -1][:20].tolist()
        _label_sum = int(_flat[_flat > -1].sum())
    else:
        _raw_shape, _raw_valid, _nonpad, _label_sum = None, 0, [], 0
    print(f"[DIAG-RAW] rank={_rank} labels_shape={_raw_shape} "
          f"valid_tokens={_raw_valid} label_sum={_label_sum} "
          f"first20_labels={_nonpad}")


def get_batch(data_iterator, is_vit_last_stage=False):
    """Generate a batch.

    When ASYNC_SCHEDULE=True and using DoubleBufferedScheduler:
      - During warmup: runs synchronously (populates group_pool cache).
      - After warmup: consumes the prefetched result from the background
        thread, then immediately kicks off prefetch for the *next* step.
    """
    global _prefetch_raw_batch, _prefetch_started

    is_async = isinstance(data_scheduler, DoubleBufferedScheduler)
    is_hybrid = hybrid_parallel is not None and hybrid_parallel == "True"

    if is_async and is_hybrid and not data_scheduler.is_warmup() and _prefetch_started:
        # ── Fast path: consume prefetched result ──
        batch = data_scheduler.swap_and_get_data()
        # EncoderBalanceComm was already applied to the raw batch before
        # prefetch, and get_data filters pixel_values — so the balanced
        # pixels are already in the result.  Just set tranfer from the
        # prefetched raw batch.
        batch['tranfer'] = _prefetch_raw_batch.get('tranfer', None) if _prefetch_raw_batch else None

        # Kick off prefetch for the NEXT step
        try:
            next_raw = _load_raw_batch(data_iterator)
            _diag_raw_batch(next_raw)
            # Apply encoder balance to raw batch BEFORE scheduler sees it
            _apply_encoder_balance(next_raw, is_vit_last_stage)
            data_scheduler.start_prefetch(next_raw)
            _prefetch_raw_batch = next_raw
            _prefetch_started = True
        except StopIteration:
            # Epoch boundary — no more data to prefetch
            _prefetch_started = False
            _prefetch_raw_batch = None

        return batch

    # ── Slow path: synchronous (non-hybrid, warmup, or first call) ──
    batch = _load_raw_batch(data_iterator)
    _diag_raw_batch(batch)
    _apply_encoder_balance(batch, is_vit_last_stage)

    if is_hybrid:
        batch = data_scheduler.next_batch(batch)

        # After warmup completes, kick off the first prefetch
        if is_async and not data_scheduler.is_warmup() and not _prefetch_started:
            try:
                next_raw = _load_raw_batch(data_iterator)
                _diag_raw_batch(next_raw)
                _apply_encoder_balance(next_raw, is_vit_last_stage)
                data_scheduler.start_prefetch(next_raw)
                _prefetch_raw_batch = next_raw
                _prefetch_started = True
            except StopIteration:
                _prefetch_started = False

    return batch


def get_tps(output_tensor):
    """Get the tokens per sample"""
    B, S, _ = output_tensor.shape
    dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    cp_size = torch.distributed.get_world_size(group=mpu.get_context_parallel_group())
    tokens_per_sample = torch.tensor(S, device=output_tensor.device) / dp_size * cp_size
    torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group())
    return tokens_per_sample


def average_losses_for_hybrid_parallel(losses, token_nums=None):
    """Token-weighted average of losses across hybrid parallel groups.

    Each CP group processes different data with potentially different token
    counts.  A simple (unweighted) mean of group losses would be biased
    towards groups with fewer tokens.  Instead, we compute the global
    token-weighted mean:

        avg_loss = Σ_groups (group_loss × group_tokens) / Σ_groups (group_tokens)

    Because multiple ranks within one CP group hold the same (loss, token_nums)
    pair, we divide each rank's contribution by its CP size before the
    all-reduce so that each *group* is counted exactly once.

    Args:
        losses: list of loss tensors (one element).
        token_nums: number of valid tokens in this group (scalar tensor).
                    If None, falls back to unweighted mean (all groups equal weight).
    """
    cp_size = torch.distributed.get_world_size(group=mpu.get_context_parallel_group())
    rank = torch.distributed.get_rank()

    if token_nums is not None:
        # --- Token-weighted average ---
        # loss_x_tokens = group_loss * group_tokens (total un-normalized loss for this group)
        # Divide by cp_size so that the cp_size ranks in one group contribute once total.
        loss_val = losses[0].clone().detach()
        tokens_val = token_nums.clone().detach().float()

        loss_x_tokens = (loss_val * tokens_val / cp_size).view(1)
        tokens_reduced = (tokens_val / cp_size).view(1)

        torch.distributed.all_reduce(loss_x_tokens)
        torch.distributed.all_reduce(tokens_reduced)

        averaged_loss = loss_x_tokens / tokens_reduced
    else:
        # Fallback: unweighted mean (same as before but explicit)
        num_groups = data_scheduler.get_num_groups()
        averaged_losses = torch.cat(
            [loss.clone().detach().view(1) / cp_size for loss in losses])
        torch.distributed.all_reduce(averaged_losses)
        averaged_loss = averaged_losses / num_groups

    return averaged_loss


def loss_func(output_tensor):
    """Loss function."""
    args = get_args()
    loss_dict = output_tensor['loss_dict']

    loss_dir = {}
    if args.log_tps:
        tokens_per_sample = get_tps(output_tensor['logits'])
        loss_dir["tokens per sample"] = tokens_per_sample

    if args.calculate_per_token_loss:
        loss, local_num_tokens, reporting_loss = compute_token_level_loss(loss_dict)
        loss_dir["loss"] = (reporting_loss[0], reporting_loss[1])
        return (
            loss[0].clone(),
            local_num_tokens,
            loss_dir
        )

    loss = loss_dict['loss']
    token_nums = loss_dict.get('token_nums', None)
    num_samples = loss_dict.get('num_samples', 1)

    is_hybrid = os.environ.get("HYBRID_PARALLEL", None) == "True"
    if is_hybrid:
        cp_size = mpu.get_context_parallel_world_size()
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)

        num_groups = mpu.get_data_parallel_world_size()
        other_ps = (mpu.get_tensor_model_parallel_world_size() *
                    mpu.get_pipeline_model_parallel_world_size())
        num_static_groups = (torch.distributed.get_world_size() //
                             other_ps // args.context_parallel_size)
        inflated_mbs = args.micro_batch_size * num_static_groups
        mbs_per_group = inflated_mbs // num_groups

        # Rescale: raw loss is mean over this group's num_samples.
        # Multiply by (num_samples / mbs_per_group) so each sample contributes
        # equal gradient weight regardless of BFD group size.
        loss = loss * (num_samples / mbs_per_group)

        # Logging: global mean = sum(loss_sum_g / MBS) / num_groups
        weighted_loss = loss.clone().detach() / cp_size
        averaged_loss = weighted_loss.view(1)
        torch.distributed.all_reduce(averaged_loss, group=dp_cp_group)
        averaged_loss = averaged_loss / num_groups
    else:
        averaged_loss = average_losses_across_data_parallel_group([loss])

    loss_dir["loss"] = averaged_loss[0]
    loss = loss.unsqueeze(0).clone()

    # === DIAG: loss_func BSND/default path (all ranks) ===
    _diag_rank = torch.distributed.get_rank()
    _token_info = f" token_nums={token_nums.item():.0f}" if token_nums is not None else ""
    _ns_info = f" num_samples={num_samples}"
    print(f"[DIAG-LOSSFUNC] rank={_diag_rank} path=BSND_default "
          f"raw_loss={loss_dict['loss'].item():.6f} "
          f"logging_loss={averaged_loss[0].item():.6f} "
          f"loss_for_backward={(loss / mpu.get_context_parallel_world_size()).squeeze().item():.6f}"
          f"{_token_info}{_ns_info}")
    # === END DIAG ===

    return loss / mpu.get_context_parallel_world_size(), loss_dir


def forward_step(data_iterator, model):
    """Forward step."""
    is_vit_last_stage = False
    if model.module.module.add_image_encoder:
        is_vit_last_stage = model.module.module.image_encoder.post_process
    output_tensor = model(**get_batch(data_iterator, is_vit_last_stage))
    return output_tensor, loss_func


def _get_img_processor_from_dataset(dataset):
    """Extract img_video_processor from a dataset (handles ConcatDataset)."""
    if hasattr(dataset, 'img_video_processor'):
        return dataset.img_video_processor
    if hasattr(dataset, 'datasets') and len(dataset.datasets) > 0:
        return _get_img_processor_from_dataset(dataset.datasets[0])
    return None


def _set_skip_pixel_values(dataset, value=True):
    """Recursively enable lightweight mode on MultiModalChatDataset instances."""
    if hasattr(dataset, 'skip_pixel_values'):
        dataset.skip_pixel_values = value
    if hasattr(dataset, 'datasets'):
        for d in dataset.datasets:
            _set_skip_pixel_values(d, value)


def _make_pixel_loader(img_processor):
    """Build a pixel_loader callback for the Scheduler's two-phase loading."""
    def pixel_loader(databatch, local_data_ids):
        image_paths_all = databatch.get('_image_path', [])  # list[N]: each item is list of paths
        image_modes_all = databatch.get('_image_mode', [])   # list[N]: mode string per sample

        all_pixel_values = []
        all_flags = []

        for idx in local_data_ids:
            paths = image_paths_all[idx]   # always a list
            mode = image_modes_all[idx] if image_modes_all else 'single_image'

            if mode == 'single_image':
                pv = img_processor(image_path=paths[0], mode='single_image', num_image=1)['pixel_values']
            elif mode == 'multi_image':
                pv_parts = []
                num_images = len(paths)
                for p in paths:
                    cur = img_processor(image_path=p, mode='multi_image', num_image=num_images)['pixel_values']
                    pv_parts.extend(cur)
                pv = torch.stack(pv_parts)
            elif mode == 'video':
                pv = img_processor(video_path=paths[0])['pixel_values']
            else:
                raise ValueError(f"Unknown image mode: {mode}")

            num_patches = pv.shape[0]
            all_pixel_values.append(pv)
            all_flags.extend([1] * num_patches)

        if all_pixel_values:
            pixel_values = torch.cat(all_pixel_values, dim=0)
        else:
            pixel_values = None
        image_flags = torch.tensor(all_flags, dtype=torch.long)
        return pixel_values, image_flags

    return pixel_loader


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    if args.hetero_parallel:
        print_rank_0("change parallel state for data loader ...")
        change_parallel_state("text_decoder")

        if args.hetero_encoder_mbs_scale > 1:
            pp_mbs = args.micro_batch_size
            args.micro_batch_size = pp_mbs * args.hetero_encoder_mbs_scale

    datasets = build_mm_dataset(data_config.dataset_param)

    # Save original MBS before any temporary modifications
    micro_batch_size = args.micro_batch_size

    if hybrid_parallel is not None and hybrid_parallel == "True":
        # Hybrid Parallel: every rank loads the same full batch, Scheduler distributes.
        # Use a dummy process_group with size=1 so BaseRandomBatchSampler generates
        # the same global permutation on all ranks (num_replicas=1, rank=0).
        class _SingleRankGroup:
            def size(self): return 1
            def rank(self): return 0

        other_ps = mpu.get_tensor_model_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size()
        num_groups = torch.distributed.get_world_size() // other_ps // args.context_parallel_size
        # Temporarily enlarge MBS so DataLoader yields enough samples for all CP groups
        args.micro_batch_size = micro_batch_size * num_groups
        process_group = _SingleRankGroup()

        # Two-phase loading: lightweight mode avoids loading pixel_values on all ranks.
        # Each rank loads only the images assigned to it by the Scheduler.
        _set_skip_pixel_values(datasets, value=True)
        img_processor = _get_img_processor_from_dataset(datasets)
        if img_processor is not None and data_scheduler is not None:
            data_scheduler.set_pixel_loader(_make_pixel_loader(img_processor))
            print_rank_0("[HybridParallel] Two-phase image loading enabled: pixel_values loaded per-rank after scheduling.")
        else:
            print_rank_0("[HybridParallel] Warning: could not enable two-phase image loading (no img_processor or scheduler).")
    else:
        process_group = mpu.get_data_parallel_group()

    build_dataloader = partial(
        build_mm_dataloader,
        dataloader_param=data_config.dataloader_param,
        process_group=process_group,
        dataset_param=data_config.dataset_param,
        consumed_samples=args.consumed_train_samples
    )
    if args.use_data_balance:
        global_batch_size = args.micro_batch_size * get_num_microbatches()
        if args.hetero_encoder_mbs_scale > 1:
            global_batch_size = global_batch_size // args.hetero_encoder_mbs_scale
        args.micro_batch_size = global_batch_size

    if isinstance(datasets, tuple) and len(datasets) == 2:
        train_dataset, valid_dataset = datasets
        train_dataloader = build_dataloader(train_dataset)
        args.micro_batch_size = micro_batch_size
        valid_dataloader = build_dataloader(valid_dataset)
        train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
    else:
        train_dataset = datasets
        val_rate = getattr(data_config.dataset_param.basic_parameters, 'val_rate', 0.0)
        if not (0.0 <= val_rate <= 1.0):
            raise ValueError(f'val_rate must be between 0.0 and 1.0, got {val_rate}')
        if isinstance(train_dataset, Dataset) and val_rate > 0:
            dataset = train_dataset.train_test_split(test_size=val_rate, seed=args.seed)
            train_dataset, valid_dataset = dataset['train'], dataset['test']
            train_dataloader = build_dataloader(train_dataset)
            args.micro_batch_size = micro_batch_size
            valid_dataloader = build_dataloader(valid_dataset)
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
        else:
            train_dataloader = build_dataloader(train_dataset)
            args.micro_batch_size = micro_batch_size
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader)

    if args.hetero_parallel and args.hetero_encoder_mbs_scale > 1:
        args.micro_batch_size = pp_mbs

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    from mindspeed_mm.patchs import ring_attn_patch, ulysses_patches, torch_dcp_patch
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external"},
    )
