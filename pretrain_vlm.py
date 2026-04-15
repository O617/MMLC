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
from hybrid_dataloader import (
    HybridScheduledDataLoader,
    set_skip_pixel_values,
    get_img_processor_from_dataset,
    make_pixel_loader,
)
from synthetic_dataloader import build_synthetic_dataloader
from static_pack_dataloader import build_static_pack_dataloader
from profiler import MLLMProfiler
mindspeed_args = get_mindspeed_args()
data_scheduler = None
hybrid_parallel = os.environ.get("HYBRID_PARALLEL")
_hybrid_loader = None  # HybridScheduledDataLoader instance (set in train_valid_test_datasets_provider)
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
    # STATIC_PACK baseline (mirror of Qwen3VL path): fixed CP×DP topology, no
    # Scheduler needed.  Megatron's --context-parallel-size handles CP; the
    # static_pack dataloader produces TND packed batches directly.
    if os.environ.get("STATIC_PACK", "").lower() == "true":
        return model

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


def get_batch(data_iterator, is_vit_last_stage=False):
    """Generate a batch."""
    if _hybrid_loader is not None:
        # Hybrid mode: all scheduling and two-phase image loading is handled
        # by HybridScheduledDataLoader.  We only propagate is_vit_last_stage
        # (a static property of the model, cheap to set each step) so that
        # EncoderBalanceComm fires on the correct pipeline stage.
        _hybrid_loader.set_vit_last_stage(is_vit_last_stage)
        return next(data_iterator)

    # ── Non-hybrid path ──────────────────────────────────────────────
    batch = _load_raw_batch(data_iterator)
    _apply_encoder_balance(batch, is_vit_last_stage)

    # STATIC_PACK baseline: dataloader yields a `seqlens` field (sub-seq
    # lengths incl. tail padding) instead of constructed PackedSeqParams.
    # Build the PackedSeqParams here so VLMModel's TND path consumes it
    # identically to the Hybrid Scheduler-produced batches.
    #
    # For the Ulysses CP + packed path, MindSpeed's patched rotary uses
    # ``cu_seqlens_q`` and its per-sample CP split utilities use
    # ``cu_seqlens_q_padded``.  The static_pack dataloader already pads every
    # sub-seq to ``2 × cp_size`` so both tensors share the same cumulative
    # layout — we pass the same cu tensor as both fields, mirroring the
    # Hybrid scheduler's convention (see scheduler.py:759).
    if 'seqlens' in batch:
        from megatron.core.packed_seq_params import PackedSeqParams
        seqlens = batch.pop('seqlens')
        if isinstance(seqlens, torch.Tensor):
            seqlens = seqlens.to(torch.int32)
        else:
            seqlens = torch.tensor(seqlens, dtype=torch.int32)
        cu = torch.zeros(seqlens.numel() + 1, dtype=torch.int32, device=seqlens.device)
        cu[1:] = torch.cumsum(seqlens, dim=0)
        cu = cu.to(device=torch.cuda.current_device())
        max_seqlen = int(seqlens.max().item())
        batch['packed_seq_params'] = PackedSeqParams(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            qkv_format='thd',
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

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
    is_static_pack = os.environ.get("STATIC_PACK", "").lower() == "true"
    if is_hybrid or is_static_pack:
        # Same SUM ÷ num_groups reduction for both Hybrid (BFD-dynamic CP
        # groups, where mpu.get_context_parallel_world_size() varies per rank)
        # and Static FFD baseline (uniform CP).  See Phase 15 in DEBUG_REPORT
        # for the derivation: each rank's loss gets rescaled by
        # num_samples/mbs_per_group, divided by its own (possibly dynamic)
        # cp_size, then SUM-reduced across the world group.  The cp_size_g
        # factor cancels because each dynamic group of size cp_size_g
        # contributes cp_size_g identical copies, yielding
        # `(Σ_g per_sample_sum_g) / mbs_per_group`, which divided by
        # num_groups gives the canonical "mean over all samples in the GBS".
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
        # equal weight regardless of BFD group size or static bucket size.
        # No-op when num_samples == mbs_per_group.
        loss = loss * (num_samples / mbs_per_group)

        weighted_loss = loss.clone().detach() / cp_size
        averaged_loss = weighted_loss.view(1)
        torch.distributed.all_reduce(averaged_loss, group=dp_cp_group)
        averaged_loss = averaged_loss / num_groups
    else:
        averaged_loss = average_losses_across_data_parallel_group([loss])

    loss_dir["loss"] = averaged_loss[0]
    loss = loss.unsqueeze(0).clone()

    return loss / mpu.get_context_parallel_world_size(), loss_dir


def forward_step(data_iterator, model):
    """Forward step."""
    is_vit_last_stage = False
    if model.module.module.add_image_encoder:
        is_vit_last_stage = model.module.module.image_encoder.post_process
    output_tensor = model(**get_batch(data_iterator, is_vit_last_stage))
    return output_tensor, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    global _hybrid_loader
    args = get_args()
    data_config = args.mm.data
    if args.hetero_parallel:
        print_rank_0("change parallel state for data loader ...")
        change_parallel_state("text_decoder")

        if args.hetero_encoder_mbs_scale > 1:
            pp_mbs = args.micro_batch_size
            args.micro_batch_size = pp_mbs * args.hetero_encoder_mbs_scale

    use_synthetic = os.environ.get("SYNTHETIC_DATA", "").lower() == "true"
    use_static_pack = os.environ.get("STATIC_PACK", "").lower() == "true"
    # Save original MBS up-front because both static_pack and hybrid branches
    # temporarily inflate it for the inner SyntheticDataLoader.
    micro_batch_size = args.micro_batch_size

    if use_static_pack:
        # Static-CP TND baseline.  Same recipe as pretrain_transformers.py:
        # share SyntheticDataLoader (so per-step B=MBS×DP samples are byte-
        # identical to the Hybrid burst run under the same seed) but assign
        # samples to ranks via a deterministic FFD bin-packing into DP buckets,
        # each capped at max_pack_seqlen=args.seq_length.  No Scheduler, no BFD.
        cp_size = args.context_parallel_size
        other_ps = (mpu.get_tensor_model_parallel_world_size()
                    * mpu.get_pipeline_model_parallel_world_size())
        dp_size = torch.distributed.get_world_size() // other_ps // cp_size
        dp_rank = mpu.get_data_parallel_rank()
        inflated_mbs = micro_batch_size * dp_size
        args.micro_batch_size = inflated_mbs
        train_dataloader = build_static_pack_dataloader(
            inflated_batch_size=inflated_mbs,
            dp_size=dp_size,
            dp_rank=dp_rank,
            cp_size=cp_size,
            max_pack_seqlen=args.seq_length,
        )
        args.micro_batch_size = micro_batch_size
        print_rank_0(
            f"[StaticPack] InternVL3 static-CP TND baseline enabled — "
            f"CP={cp_size}, DP={dp_size}, MBS_user={micro_batch_size}, "
            f"inflated_B={inflated_mbs}, seq_length={args.seq_length}, "
            f"min_len={os.environ.get('SYNTHETIC_MIN_LEN', 512)}, "
            f"max_len={os.environ.get('SYNTHETIC_MAX_LEN', 8192)}, "
            f"dist={os.environ.get('SYNTHETIC_LENGTH_DIST', 'uniform')}, "
            f"num_batches={os.environ.get('SYNTHETIC_NUM_BATCHES', 1000)}"
        )
        train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader)
        return train_dataloader, valid_dataloader, test_dataloader

    if not use_synthetic:
        datasets = build_mm_dataset(data_config.dataset_param)

    is_hybrid = hybrid_parallel is not None and hybrid_parallel == "True"

    if is_hybrid:
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

        if not use_synthetic:
            # Two-phase loading: lightweight mode avoids loading pixel_values on all ranks.
            # Each rank loads only the images assigned to it by the Scheduler.
            set_skip_pixel_values(datasets, value=True)
            img_processor = get_img_processor_from_dataset(datasets)
            if img_processor is not None and data_scheduler is not None:
                data_scheduler.set_pixel_loader(make_pixel_loader(img_processor))
                print_rank_0("[HybridParallel] Two-phase image loading enabled: pixel_values loaded per-rank after scheduling.")
            else:
                print_rank_0("[HybridParallel] Warning: could not enable two-phase image loading (no img_processor or scheduler).")
    else:
        process_group = mpu.get_data_parallel_group()

    # Build raw dataloaders (build_iterations is called after optional wrapping).
    valid_dataloader = None
    if use_synthetic:
        train_dataloader = build_synthetic_dataloader(args.micro_batch_size)
        _with_vis = os.environ.get('SYNTHETIC_WITH_VISION', 'false').lower() == 'true'
        print_rank_0(
            f"[SyntheticData] Synthetic dataloader enabled — "
            f"batch_size={args.micro_batch_size}, "
            f"min_len={os.environ.get('SYNTHETIC_MIN_LEN', 512)}, "
            f"max_len={os.environ.get('SYNTHETIC_MAX_LEN', 8192)}, "
            f"num_batches={os.environ.get('SYNTHETIC_NUM_BATCHES', 1000)}, "
            f"with_vision={_with_vis}"
            + (f", vision_ratio={os.environ.get('SYNTHETIC_VISION_RATIO', 0.8)}, "
               f"tile_tokens={os.environ.get('SYNTHETIC_TILE_TOKENS', 256)}"
               if _with_vis else "")
        )
        args.micro_batch_size = micro_batch_size
    else:
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
            else:
                train_dataloader = build_dataloader(train_dataset)
                args.micro_batch_size = micro_batch_size

    # Wrap train_dataloader with HybridScheduledDataLoader before build_iterations.
    # build_iterations wraps dataloaders in a cyclic generator; calling next() on
    # that generator transparently calls HybridScheduledDataLoader.__next__(), which
    # runs the Scheduler and two-phase image loading for each step.
    if is_hybrid:
        _hybrid_loader = HybridScheduledDataLoader(
            base_loader=train_dataloader,
            scheduler=data_scheduler,
            float_dtype=args.params_dtype,
            encoder_dp_balance=getattr(args, 'encoder_dp_balance', False),
        )
        train_dataloader = _hybrid_loader

    if valid_dataloader is not None:
        train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
    else:
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
