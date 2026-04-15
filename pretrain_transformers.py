# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain VLM (ViT+MLP+LLM) MODEL."""
from copy import deepcopy
from functools import partial
from typing import Dict, Any
import importlib.util
import os

from datasets import Dataset
import torch
import transformers
from packaging import version

# Patch ALL possible locations BEFORE any transformers import
if version.parse(transformers.__version__).major >= 5:
    def _dummy_check_model_inputs(*args, **kwargs):
        """
        Universal dummy for @check_model_inputs.
        Supports both:
        - @check_model_inputs          → called as check_model_inputs(cls)
        - @check_model_inputs(...)     → called as check_model_inputs(...)(cls)
        """
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # Case 1: @check_model_inputs (no parentheses) → args[0] is the class/function
            return args[0]
        else:
            # Case 2: @check_model_inputs(...) → return a decorator that returns the function
            def decorator(fn):
                return fn
            return decorator

    import transformers.utils.generic
    transformers.utils.generic.check_model_inputs = _dummy_check_model_inputs

spec = importlib.util.spec_from_file_location("config_loader", "mindspeed_mm/configs/read_yaml_config.py")
spec.loader.exec_module(importlib.util.module_from_spec(spec))
import mindspeed.megatron_adaptor
from mindspeed.megatron_adaptor import get_mindspeed_args
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import average_losses_across_data_parallel_group, unwrap_model
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations, cal_gradient_accumulation_size
from mindspeed_mm.data.data_utils.constants import AVG_PER_STEP_TOKEN_NUM, GLOBAL_STEP_TOKEN_NUM
from mindspeed_mm.data.dataloader.dataloader import PrefetchGradAccDataLoader
from mindspeed_mm.data.dataloader.dynamic_batching_dataloader import DynamicBatchingDataLoader
from mindspeed_mm.training import pretrain
from mindspeed_mm.models.transformers_model import TransformersModel
from scheduler import Scheduler, DoubleBufferedScheduler
from hybrid_dataloader import (
    HybridScheduledDataLoader,
    set_skip_pixel_values,
    get_img_processor_from_dataset,
    make_pixel_loader,
)
from synthetic_dataloader import build_synthetic_dataloader
from static_pack_dataloader import build_static_pack_dataloader

mindspeed_args = get_mindspeed_args()
if hasattr(mindspeed_args, "ai_framework") and mindspeed_args.ai_framework == "mindspore" and mindspeed_args.optimization_level >= 0:
    import mindspeed_mm.mindspore.mindspore_adaptor

hybrid_parallel = os.environ.get("HYBRID_PARALLEL")
_hybrid_loader = None   # HybridScheduledDataLoader instance (set in train_valid_test_datasets_provider)
data_scheduler = None   # Scheduler instance (set in model_provider)


def model_provider(*args, **kwargs):
    """Builds the model."""
    args = get_args()
    print_rank_0("building VLMModel ...")
    vlm_config = deepcopy(args.mm.model)
    model = TransformersModel(vlm_config)

    global data_scheduler
    # STATIC_PACK baseline: fixed CP × DP topology, no Scheduler needed.
    # Megatron's --context-parallel-size handles CP; the static_pack dataloader
    # produces TND packed batches directly, so we skip Scheduler construction
    # entirely (avoids any chance of its side effects affecting mpu groups).
    if os.environ.get("STATIC_PACK", "").lower() == "true":
        return model

    other_parallel_group_size = (mpu.get_tensor_model_parallel_world_size()
                                 * mpu.get_pipeline_model_parallel_world_size())
    use_async = os.environ.get("ASYNC_SCHEDULE", "False") == "True"
    scheduler_cls = DoubleBufferedScheduler if use_async else Scheduler
    # seq_len_chunk=8192 for Qwen3VL so ceil(256k/chunk)=32 covers the largest
    # supported burst; must stay in lockstep with SYNTHETIC_TOKEN_BUDGET_PER_GPU
    # in the launch script so the producer never exceeds the cluster-wide budget.
    #
    # max_cp_degree == text-decoder num_attention_heads: Ulysses splits Q heads
    # across CP ranks, so CP > num_attention_heads is invalid.  Read directly
    # from the HF config (2B→16, 4B→32).
    text_cfg = getattr(model.transformer_config, 'text_config', model.transformer_config)
    num_attention_heads = getattr(text_cfg, 'num_attention_heads', 32)
    data_scheduler = scheduler_cls(
        cluster_size=torch.distributed.get_world_size(),
        other_parallel_group_size=other_parallel_group_size,
        img_context_token_id=model.img_context_token_id,
        cp_window_size=get_args().context_parallel_size,
        schedule_mode=os.environ.get("SCHEDULE_MODE", "dynamic"),
        seq_len_chunk=int(os.environ.get("SCHED_SEQ_LEN_CHUNK", 8192)),
        max_cp_degree=num_attention_heads,
        # Ulysses splits Q heads across CP ranks, so inflation must keep CP as
        # a power of two to stay a divisor of num_attention_heads (16/32).
        cp_must_be_power_of_two=True,
    )

    if hybrid_parallel is not None and hybrid_parallel == "True":
        from megatron.core.num_microbatches_calculator import reconfigure_num_microbatches_calculator
        other_ps = (mpu.get_tensor_model_parallel_world_size()
                    * mpu.get_pipeline_model_parallel_world_size())
        num_groups = torch.distributed.get_world_size() // other_ps // args.context_parallel_size
        reconfigure_num_microbatches_calculator(
            rank=torch.distributed.get_rank(),
            rampup_batch_size=None,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            data_parallel_size=num_groups,
        )

    return model


def move_to_device(batch: Dict[str, Any], float_dtype: str):
    """Move batch tensors to current device with given float dtype."""
    from megatron.core.packed_seq_params import PackedSeqParams
    new_batch = dict()
    for k, v in batch.items():
        if k in [AVG_PER_STEP_TOKEN_NUM, GLOBAL_STEP_TOKEN_NUM]:
            new_batch[k] = v.to(device=torch.cuda.current_device())
        elif isinstance(v, PackedSeqParams):
            # Move all tensor fields of PackedSeqParams to device in-place; keep object reference.
            dev = torch.cuda.current_device()
            for field in ('cu_seqlens_q', 'cu_seqlens_kv', 'cu_seqlens_q_padded',
                          'cu_seqlens_kv_padded', 'max_seqlen_q', 'max_seqlen_kv'):
                t = getattr(v, field, None)
                if isinstance(t, torch.Tensor):
                    setattr(v, field, t.to(device=dev))
            new_batch[k] = v
        elif isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            new_batch[k] = v.to(device=torch.cuda.current_device(), dtype=dtype)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            new_batch[k] = [t.to(device=torch.cuda.current_device(),
                             dtype=float_dtype if torch.is_floating_point(t) else None)
                        for t in v]
        elif isinstance(v, (bool, int, float, str)) or v is None:
            new_batch[k] = v
    return new_batch


def get_batch(data_iterator):
    """Generate a batch."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")

    if _hybrid_loader is not None:
        # Hybrid Parallel: scheduler produces TND packed batches.
        # TransformersModel._compute_tnd_loss() consumes packed_seq_params — keep it.
        # image_flags is synthesised by the scheduler but is not a valid HF model input.
        batch.pop('image_flags', None)

    # STATIC_PACK baseline: dataloader yields a `seqlens` field (sub-seq lengths
    # incl. tail padding) instead of constructed PackedSeqParams.  Build the
    # PackedSeqParams here so the model TND path consumes it identically to
    # the Hybrid Scheduler-produced batches.
    if 'seqlens' in batch:
        from megatron.core.packed_seq_params import PackedSeqParams
        seqlens = batch.pop('seqlens').to(torch.int32)
        cu = torch.zeros(seqlens.numel() + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(seqlens, dim=0)
        max_seqlen = int(seqlens.max().item())
        batch['packed_seq_params'] = PackedSeqParams(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            qkv_format='thd',
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

    return batch


def loss_func(output_tensor):
    """Loss function.

    TransformersModel produces a per-rank loss of ``local_sum / global_count``
    (see ``build_loss_ctx``/``_compute_tnd_loss``): each CP rank holds a
    ``1/cp_size`` fraction of the full group mean, and summing across CP ranks
    recovers it.  This differs from ``VLMModel``, whose CP path gathers back to
    the full loss on every rank.

    Consequences for this function:

    * **Backward**: ``schedules.py`` multiplies the returned loss by
      ``cp_size / num_microbatches`` (see ``schedules.py:297-298``).  We divide
      by ``cp_size`` before returning so that per-rank backward loss ends up
      at ``local_sum / (global_count * num_microbatches)``; summing that
      across CP ranks gives ``full_loss / num_microbatches``.

    * **Logging**: since each rank holds a fraction, ``AVG`` across the
      dp+cp group followed by ``* cp_size`` yields the DP-average of per-group
      full means.

    * **Hybrid (BFD dynamic)**: different DP groups may receive different
      sample counts.  We scale the raw loss by ``num_samples / mbs_per_group``
      (mirroring ``pretrain_vlm.loss_func``) so that each sample contributes
      equal gradient weight regardless of group size.  For naive scheduling
      ``num_samples == mbs_per_group`` and the factor is 1.
    """
    args = get_args()
    loss_dir = {}
    cp_size = mpu.get_context_parallel_world_size()

    loss = output_tensor['loss']
    if output_tensor.get('token_nums', None) is not None:
        total_tokens = output_tensor['token_nums']
    else:
        loss_mask = output_tensor['loss_mask'].view(-1).float()
        total_tokens = loss_mask.sum()
    num_samples = output_tensor.get('num_samples', 1)

    if args.log_tps:
        dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        tokens_per_sample = torch.tensor(total_tokens / args.micro_batch_size, device=output_tensor['loss'].device) / dp_size
        torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group(with_context_parallel=True))
        loss_dir["tokens per sample"] = tokens_per_sample

    is_hybrid = hybrid_parallel is not None and hybrid_parallel == "True"
    is_static_pack = os.environ.get("STATIC_PACK", "").lower() == "true"
    mbs_per_group = None
    if is_hybrid or is_static_pack:
        # Same rescale formula for both Hybrid (BFD-dynamic CP groups) and
        # Static FFD baseline: ranks may hold a different number of samples
        # (e.g. the burst-bearing bucket has 1 sample while peers hold ~21),
        # so weighting each rank's loss by num_samples/mbs_per_group equalises
        # per-sample gradient contributions and makes the cross-DP+CP reduction
        # produce the correct overall per-sample mean.
        num_groups = mpu.get_data_parallel_world_size()
        other_ps = (mpu.get_tensor_model_parallel_world_size() *
                    mpu.get_pipeline_model_parallel_world_size())
        num_static_groups = (torch.distributed.get_world_size() //
                             other_ps // args.context_parallel_size)
        inflated_mbs = args.micro_batch_size * num_static_groups
        mbs_per_group = inflated_mbs // num_groups

        # No-op when num_samples == mbs_per_group (e.g. SCHEDULE_MODE=naive,
        # or non-burst microbatches in the static FFD path).
        loss = loss * (num_samples / mbs_per_group)

    # Reduce the per-rank local-fraction loss into a single per-step mean.
    #
    # Each rank holds  loss_local = (per_sample_mean_g / cp_size_g) ×
    # (num_samples_g / mbs_per_group)  where cp_size_g is THIS rank's current
    # CP group size (which can vary across ranks under Hybrid's dynamic CP
    # scheduler).  Summing across all WORLD ranks collapses the cp_size_g
    # factor (each dynamic group of size cp_size_g contributes cp_size_g
    # identical copies, and cp_size_g cancels out), leaving:
    #
    #     Σ_ranks loss_local = Σ_g (per_sample_mean_g × num_samples_g / mbs_per_group)
    #                        = (Σ_g per_sample_sum_g) / mbs_per_group
    #                        = total_loss / mbs_per_group
    #
    # The canonical metric we want is total_loss / total_samples, where
    # total_samples = mbs_per_group × num_groups = inflated_mbs.  So we still
    # need to divide by num_groups to drop the mbs_per_group residue and
    # multiply by 1/num_groups in one step.  This formulation is invariant to
    # whichever dynamic CP group rank 0 happens to belong to, fixing the
    # rank-dependent logged values the old AVG×cp_size formula produced under
    # Hybrid.
    averaged_loss = loss.clone().detach().view(1)
    if is_hybrid or is_static_pack:
        torch.distributed.all_reduce(
            averaged_loss,
            group=mpu.get_data_parallel_group(with_context_parallel=True),
            op=torch.distributed.ReduceOp.SUM,
        )
        averaged_loss /= float(num_groups)
    else:
        # Non-hybrid baseline path: uniform static CP, original convention.
        torch.distributed.all_reduce(
            averaged_loss,
            group=mpu.get_data_parallel_group(with_context_parallel=True),
            op=torch.distributed.ReduceOp.AVG,
        )
        averaged_loss *= cp_size
    loss_dir["loss"] = averaged_loss[0]

    loss = loss.unsqueeze(0).clone()
    return loss / cp_size, loss_dir


def forward_step(data_iterator, model):
    """Forward step."""
    batch_data = get_batch(data_iterator)
    if get_args().use_torch_fsdp2:
        from mindspeed_mm.tasks.finetune.lora.utils import is_enable_lora
        model_unwrapped = unwrap_model(model)
        fsdp_core_model = model_unwrapped.model.model if is_enable_lora() else model_unwrapped.model
        dtype = fsdp_core_model._get_fsdp_state()._mp_policy.param_dtype
        dtype = dtype if dtype is not None else torch.bfloat16
        batch_data = move_to_device(batch_data, dtype)
    else:
        batch_data = move_to_device(batch_data, get_args().params_dtype)

    output_tensor = model(**batch_data)
    return output_tensor, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    global _hybrid_loader
    args = get_args()
    data_config = args.mm.data

    use_synthetic = os.environ.get("SYNTHETIC_DATA", "").lower() == "true"
    use_static_pack = os.environ.get("STATIC_PACK", "").lower() == "true"
    # Save original MBS before any temporary modifications.  The STATIC_PACK
    # and Hybrid branches both temporarily inflate it for the inner loader.
    micro_batch_size = args.micro_batch_size
    if use_static_pack:
        # Static-CP TND baseline.  Reuses SyntheticDataLoader (so the per-step
        # B=MBS×DP samples are byte-identical to the Hybrid burst run under
        # the same seed) but assigns samples to ranks via deterministic FFD
        # bin-packing into DP buckets, each capped at max_pack_seqlen.  No
        # Scheduler, no BFD repartitioning — that is the entire point of the
        # comparison.
        cp_size = args.context_parallel_size
        other_ps = (mpu.get_tensor_model_parallel_world_size()
                    * mpu.get_pipeline_model_parallel_world_size())
        dp_size = torch.distributed.get_world_size() // other_ps // cp_size
        dp_rank = mpu.get_data_parallel_rank()
        # Inflate so the inner SyntheticDataLoader yields B = MBS_user × DP
        # samples per microbatch, matching the Hybrid path's loader call.
        inflated_mbs = micro_batch_size * dp_size
        args.micro_batch_size = inflated_mbs
        train_dataloader = build_static_pack_dataloader(
            inflated_batch_size=inflated_mbs,
            dp_size=dp_size,
            dp_rank=dp_rank,
            cp_size=cp_size,
            max_pack_seqlen=args.seq_length,
        )
        args.micro_batch_size = micro_batch_size  # restore
        print_rank_0(
            f"[StaticPack] Qwen3VL static-CP TND baseline enabled — "
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
        # Hybrid Parallel: every rank loads the same full batch (num_replicas=1).
        class _SingleRankGroup:
            def size(self): return 1
            def rank(self): return 0

        other_ps = (mpu.get_tensor_model_parallel_world_size()
                    * mpu.get_pipeline_model_parallel_world_size())
        num_groups = torch.distributed.get_world_size() // other_ps // args.context_parallel_size
        # Temporarily enlarge MBS so DataLoader yields enough samples for all CP groups
        args.micro_batch_size = micro_batch_size * num_groups
        process_group = _SingleRankGroup()

        if not use_synthetic:
            # Two-phase image loading: for multimodal datasets with img_video_processor,
            # pixel_values are loaded only for the assigned samples after scheduling.
            # For huggingface dataset type (Qwen3VL), this is a no-op (img_processor=None).
            set_skip_pixel_values(datasets, value=True)
            img_processor = get_img_processor_from_dataset(datasets)
            if img_processor is not None and data_scheduler is not None:
                data_scheduler.set_pixel_loader(make_pixel_loader(img_processor))
                print_rank_0("[HybridParallel] Two-phase image loading enabled.")
            else:
                print_rank_0("[HybridParallel] Warning: could not enable two-phase image loading "
                             "(no img_processor or scheduler). All ranks will load full pixel_values.")
    else:
        process_group = mpu.get_data_parallel_group()

    valid_dataloader = None
    if use_synthetic:
        train_dataloader = build_synthetic_dataloader(args.micro_batch_size)
        _with_vis = os.environ.get('SYNTHETIC_WITH_VISION', 'false').lower() == 'true'
        print_rank_0(
            f"[SyntheticData] Qwen3VL synthetic dataloader enabled — "
            f"batch_size={args.micro_batch_size}, "
            f"min_len={os.environ.get('SYNTHETIC_MIN_LEN', 512)}, "
            f"max_len={os.environ.get('SYNTHETIC_MAX_LEN', 8192)}, "
            f"num_batches={os.environ.get('SYNTHETIC_NUM_BATCHES', 1000)}, "
            f"with_vision={_with_vis}"
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
                if args.use_txt_dynamic_batching:
                    train_dataloader = DynamicBatchingDataLoader(
                        train_dataloader,
                        max_seq_len=args.max_seq_len,
                        dynamic_batch_buffer_size=args.dynamic_batch_buffer_size,
                        vision_layout=args.mm.model.image_encoder.vision_encoder.attn_layout,
                        consumed_train_samples=args.consumed_train_samples,
                    )
            else:
                train_dataloader = build_dataloader(train_dataset)
                args.micro_batch_size = micro_batch_size
                if args.use_txt_dynamic_batching:
                    train_dataloader = DynamicBatchingDataLoader(
                        train_dataloader,
                        max_seq_len=args.max_seq_len,
                        dynamic_batch_buffer_size=args.dynamic_batch_buffer_size,
                        vision_layout=args.mm.model.image_encoder.vision_encoder.attn_layout,
                        consumed_train_samples=args.consumed_train_samples,
                    )

    # Wrap train_dataloader with HybridScheduledDataLoader before build_iterations.
    if is_hybrid:
        _hybrid_loader = HybridScheduledDataLoader(
            base_loader=train_dataloader,
            scheduler=data_scheduler,
            float_dtype=args.params_dtype,
            encoder_dp_balance=False,  # TransformersModel manages ViT internally
        )
        train_dataloader = _hybrid_loader

    if valid_dataloader is not None:
        train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
    else:
        train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader)

    loss_config = getattr(args.mm.model, "loss_cfg", None)
    use_prefetch_gradacc_dataloader = False
    if loss_config:
        use_prefetch_gradacc_dataloader = (getattr(loss_config, "loss_type", "default") == "per_token_loss")
    if use_prefetch_gradacc_dataloader:
        train_dataloader = PrefetchGradAccDataLoader(train_dataloader, grad_acc_step=cal_gradient_accumulation_size())

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    from mindspeed_mm.patchs import torch_dcp_patch
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external"},
    )
