from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import tensor_parallel, mpu
from mindspeed.core.context_parallel.model_parallel_utils import (
    get_context_parallel_group_for_hybrid_ulysses,
    get_context_parallel_group_for_hybrid_ring,
    get_context_parallel_for_hybrid_ulysses_world_size
)

from mindspeed_mm.data.data_utils.constants import AVG_PER_STEP_TOKEN_NUM
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.chunkloss import chunk_loss, calculate_lm_loss, fixed_cross_entropy
from mindspeed_mm.models.common.communications import cal_split_sizes, split_forward_gather_backward, split_forward_gather_backward_with_cp
from mindspeed_mm.models.transformers.modelhub import ModelHub
from mindspeed_mm.utils.utils import split_forward_gather_backward_with_megatron_cp
from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import gather_forward_split_backward


class TransformersModel(MultiModalModule):
    """Transformer-based multi-modal model wrapper inherited from MultiModalModule.

    Core wrapper class for initializing, loading and running transformer-based vision-language
    multi-modal models with multiple loss calculation strategies and distributed parallel training support.
    Implements context parallel loss computation, chunk-based memory-efficient loss calculation,
    model sharding and MoE auxiliary loss for large-scale model training.

    Attributes:
        config: Core transformer model configuration parsed from global arguments.
        transformer_config: HuggingFace AutoConfig instance for the underlying transformer model.
        model: Initialized transformer multi-modal model instance.
        loss_compute_mode: Loss calculation mode, supports `default` and `chunk`.
        loss_chunk_size: Chunk size for memory-efficient chunk loss calculation (default: 1024).
        router_aux_loss_coef: Coefficient for MoE model router auxiliary loss (default: 0.0).
    """
    def __init__(self, config) -> None:
        """Initialize the TransformersModel with given configuration and load pretrained weights.

        Args:
            config: General configuration for the multi-modal transformer model,
            the configuration content is derived from model.json.
        """
        super().__init__(config=config)
        args = get_args()

        hf_path = args.mm.model.init_from_hf_path
        trust_remote_code = args.trust_remote_code
        self.config = core_transformer_config_from_args(args)
        self.transformer_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=trust_remote_code)

        model_cls = ModelHub.build(config, self.transformer_config)

        self._set_loss_cfg(args)
        
        if callable(getattr(model_cls, 'overwrite_transformer_config', None)):
            self.transformer_config = model_cls.overwrite_transformer_config(self.transformer_config)

        if args.init_model_with_meta_device:
            self.model = model_cls._from_config(self.transformer_config).float()
            for m in self.model.modules():
                if getattr(m, "_is_hf_initialized", False):
                    m._is_hf_initialized = False
        else:
            self.model = model_cls.from_pretrained(
                hf_path,
                config=self.transformer_config,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
                trust_remote_code=trust_remote_code
            )
        print_rank_0("> load model successfully")

        self.model.train()

        if callable(getattr(self.model, 'freeze', None)):
            self.model.freeze(config)

        self.model.use_cache = False

    @property
    def img_context_token_id(self):
        """Image token ID for the Hybrid Parallel scheduler.

        Reads image_token_id from the HuggingFace config (e.g. 151655 for Qwen3VL).
        Falls back to video_token_id, then None if neither is present.
        """
        tid = getattr(self.transformer_config, 'image_token_id', None)
        if tid is None:
            tid = getattr(self.transformer_config, 'video_token_id', None)
        return tid

    def forward(
            self,
            input_ids: torch.Tensor,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            packed_seq_params=None,
            *args, **kwargs
    ) -> torch.Tensor:
        loss_dict = {}

        # Safety: if the data pipeline provides pixel_values for a batch that contains no
        # image placeholder tokens (e.g., text-only sequences from a VLM collator), strip
        # them before passing to the model to avoid "image features/tokens mismatch" errors.
        if pixel_values is not None and self.img_context_token_id is not None:
            n_img_tokens = (input_ids == self.img_context_token_id).sum()
            if n_img_tokens == 0:
                pixel_values = None
                image_grid_thw = None

        # TND packed mode (Hybrid Scheduler): pass seqlens kwarg so the model can:
        #   1. split input_ids[0] for per-sequence M-RoPE via get_rope_index(sequence_length=seqlens)
        #   2. build cu_seqlens for the FlashAttention var-len sub-sequence boundaries
        # Both consumers require seqlens to sum to input_ids.shape[1] (the PADDED total).
        # Padding tokens within each sub-sequence have labels=-100, so they contribute no
        # loss; pre-compute `indices = arange(T)` so the Qwen3VL model skips its
        # attention_mask-based unpad filter (which would otherwise shrink the token count
        # below the padded seqlens and mismatch cu_seqlens).
        if packed_seq_params is not None:
            cu = packed_seq_params.cu_seqlens_q.long()
            seqlens = (cu[1:] - cu[:-1]).to(torch.int32)
            kwargs['seqlens'] = seqlens
            total_len = input_ids.shape[1]
            kwargs['indices'] = torch.arange(total_len, device=input_ids.device)
            position_ids = None

        # aux loss (for moe model)
        if self.router_aux_loss_coef > 0.0:
            kwargs["output_router_logits"] = True

        if self.loss_compute_mode == "dynamic_chunk":
            kwargs["total_size"] = self.loss_chunk_size

        if packed_seq_params is not None:
            # TND packed path: always use default mode (chunk loss is incompatible with TND)
            if self.loss_compute_mode != "default":
                raise NotImplementedError(
                    f"TND packed format (packed_seq_params) requires loss_compute_mode='default', "
                    f"got '{self.loss_compute_mode}'"
                )
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=False,
                **kwargs
            )
            # Keep logits in their native bf16 — _compute_tnd_loss upcasts once
            # inside vocab_parallel_cross_entropy.  Doing an extra .float() here
            # would hold both bf16 and fp32 copies of a [1, T_local, vocab] tensor
            # simultaneously (≈7.5 GB peak at T_local=8k, vocab=152k), which
            # tips the burst CP group over the 61 GB NPU budget.
            logits = outputs.logits
            tnd_result = self._compute_tnd_loss(logits, labels, packed_seq_params)
            # Propagate every field from _compute_tnd_loss so loss_func sees the
            # real per-microbatch num_samples / token_nums (otherwise it falls
            # back to defaults: num_samples=1 → rescale factor cp_size× too small,
            # token_nums=loss_mask.sum() → over-counts padding positions).
            loss_dict.update(tnd_result)
        elif self.loss_compute_mode in ["chunk", "dynamic_chunk"]:
            loss_ctx, loss_mask = self.build_loss_ctx(labels, chunk_size=self.loss_chunk_size, **kwargs)
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=False,
                loss_ctx=loss_ctx,
                **kwargs
            )
            loss_dict["loss"] = outputs.loss
            loss_dict["loss_mask"] = loss_mask
        else:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=False,
                **kwargs
            )
            logits = outputs.logits.contiguous().float()

            loss_ctx, loss_mask = self.build_loss_ctx(labels, chunk_size=None, **kwargs)
            loss_dict["loss"] = loss_ctx(logits)
            loss_dict["loss_mask"] = loss_mask

        if hasattr(outputs, "aux_loss") and self.router_aux_loss_coef > 0:
            loss_dict["loss"] += self.router_aux_loss_coef * outputs.aux_loss

        return loss_dict

    def _compute_tnd_loss(self, logits: torch.Tensor, labels: torch.Tensor, packed_seq_params) -> dict:
        """Compute loss for TND packed sequences produced by the Hybrid Scheduler.

        The Scheduler concatenates assigned sub-sequences into a single packed tensor
        [1, T] with padding (0 in input_ids, -100 in labels) between sub-sequences.
        packed_seq_params.cu_seqlens_q marks the padded sub-sequence boundaries.

        Key difference from BSND: label shift must happen WITHIN each sub-sequence
        to prevent the last padding token of sub-seq i from being trained against the
        first real token of sub-seq i+1.

        Supported loss_types:
          - "default" / "token_loss": token mean
              (Σ valid_token_losses / total_valid_tokens)
          - "per_sample_loss": mean of per-subsequence mean losses

        Both branches return a per-CP-rank loss in the **local-fraction** convention
        (= 1/cp_size of the full value), so :func:`pretrain_transformers.loss_func`'s
        existing ``averaged_loss × cp_size`` step recovers the full-rank value
        without any branch-specific handling.

        Note on memory: this used to chunk the CE along the sequence dim (chunks
        of 1024) because Phase 14's burst run hit OOM materialising the full
        ``[1, T_local, vocab] → fp32`` logits.  Once ``--use-distributed-optimizer``
        is enabled the optimizer state shrinks to ``1/dp_size`` per rank, leaving
        plenty of headroom for the one-shot fp32 logits, so the chunking is
        removed.  Re-introduce it only if a future workload OOMs again.
        """
        cu_seqlens = packed_seq_params.cu_seqlens_q.long()
        num_subseqs = cu_seqlens.numel() - 1
        total_len = labels.shape[-1]
        cp_size = mpu.get_context_parallel_world_size()

        # Per-sub-sequence label shift: labels[t] = next token within the same sub-sequence.
        # The last real token position in each sub-seq gets -100 (no valid next token).
        # Padding positions (already -100 in labels) remain -100.
        shift_labels = torch.full_like(labels, -100)
        valid_tokens_per_sample = torch.zeros(num_subseqs, dtype=torch.long, device=labels.device)

        for i in range(num_subseqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            if end - start > 1:
                shift_labels[0, start:end - 1] = labels[0, start + 1:end]
            valid_tokens_per_sample[i] = int((shift_labels[0, start:end] > -1).sum())

        loss_mask = shift_labels > -1  # [1, T]

        # Single-shot CE on the full local sequence.  vocab_parallel_cross_entropy
        # expects fp32 logits; the .float() materialises a [1, T_local, vocab] fp32
        # tensor that distributed_optimizer's freed memory now comfortably fits.
        if cp_size > 1:
            shift_labels_local = split_forward_gather_backward_with_cp(shift_labels, dim=-1)
        else:
            shift_labels_local = shift_labels

        loss_local = tensor_parallel.vocab_parallel_cross_entropy(logits.float(), shift_labels_local)
        loss_local = loss_local * (shift_labels_local > -1)

        if self.loss_type in ("default", "token_loss"):
            # Token mean: Σ valid_token_losses / Σ valid_tokens.  alpha is computed
            # on the FULL shift_labels (pre-CP split) so each CP rank reports
            # local_sum / global_count → loss_func × cp_size restores the full mean.
            alpha = loss_mask.sum().clamp(min=1)
            loss = loss_local.sum() / alpha

        elif self.loss_type == "per_sample_loss":
            # Per-sub-seq token mean → mean across sub-seqs.
            #
            # Gather the local CE slice across the CP group so each rank holds the
            # full per-token loss for the bucket; gather_forward_split_backward
            # routes gradients back to the originating rank on backward.  The
            # final ``/ cp_size`` puts the result back into the local-fraction
            # convention so the loss_func × cp_size step in pretrain_transformers
            # recovers the full per-sample mean.
            sample_ids = torch.full((1, total_len), -1, dtype=torch.long, device=labels.device)
            for i in range(num_subseqs):
                s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                sample_ids[0, s:e] = i

            if cp_size > 1:
                full_loss = gather_forward_split_backward(
                    loss_local, mpu.get_context_parallel_group(), dim=-1
                )
            else:
                full_loss = loss_local

            flat_loss = full_loss.view(-1)
            flat_ids = sample_ids.view(-1)
            valid_mask = flat_ids >= 0

            per_sample_sum = torch.zeros(num_subseqs, dtype=full_loss.dtype, device=full_loss.device)
            per_sample_sum.scatter_add_(0, flat_ids[valid_mask], flat_loss[valid_mask])

            valid_tokens_f = valid_tokens_per_sample.to(full_loss.dtype).clamp(min=1.0)
            per_sample_mean = per_sample_sum / valid_tokens_f
            # Mask out empty sub-sequences (e.g. tail-padding) so they don't
            # drag the mean down to zero.
            non_empty = (valid_tokens_per_sample > 0).to(full_loss.dtype)
            n_real = non_empty.sum().clamp(min=1.0)
            loss = (per_sample_mean * non_empty).sum() / n_real

            if cp_size > 1:
                loss = loss / cp_size

        else:
            raise NotImplementedError(
                f"TND packed loss for loss_type='{self.loss_type}' is not implemented. "
                f"Supported: 'default', 'token_loss', 'per_sample_loss'."
            )

        # num_samples counts only non-tail-pad sub-sequences so loss_func's
        # rescale weight matches the per_sample_loss branch's denominator.
        num_real_samples = int((valid_tokens_per_sample > 0).sum())

        return {
            "loss": loss,
            "loss_mask": loss_mask,
            "num_samples": num_real_samples,
            "token_nums": loss_mask.sum().detach(),
        }

    def fully_shard(
        self,
        process_group,
        fsdp2_config_path,
        **kwargs
    ):
        # If the model has its own 'fully_shard' method, use it directly
        if hasattr(self.model, 'fully_shard') and callable(getattr(self.model, 'fully_shard')):
            return self.model.fully_shard(
                process_group=process_group,
                fsdp2_config_path=fsdp2_config_path,
                **kwargs
            )
        return False

    def calculate_chunk_size(self, batch_size: int, total_size: int) -> int:
        """
        Calculate dynamic Chunk Size to ensure batch_size * chunk_size ≤ total size, 
        where chunk_size is the largest power of two not exceeding the theoretical maximum value.

        Args:
            batch_size (int): Input batch size
            total_size (int): Upper limit of total tokens (batch_size * chunk_size),
                typically configured as the maximum token capacity of the device (e.g., 4096/8192 tokens).

        Returns:
            int: Dynamic Chunk Size that meets the requirements, returns 1 by default (when input is invalid)
        """
        if batch_size <= 0 or total_size <= 0:
            print_rank_0(f"[ERROR] Batch size={batch_size} or total size={total_size} must be a positive integer!")
            return 1
        if batch_size >= total_size:
            print_rank_0(f"[ERROR] Batch size={batch_size} exceeds total size={total_size}!")
            return 1

        max_possible_chunk_size = total_size // batch_size

        if max_possible_chunk_size == 0:
            print_rank_0(f"[ERROR] No valid Chunk Size for batch size batch_size={batch_size}!")
            return 1

        max_power_of_two_chunk_size = 1 << (max_possible_chunk_size.bit_length() - 1)

        if max_power_of_two_chunk_size > max_possible_chunk_size:
            max_power_of_two_chunk_size = max_power_of_two_chunk_size >> 1  # Right shift by 1 bit = divide by 2

        return max_power_of_two_chunk_size

    def build_loss_ctx(
        self,
        labels,
        ignore_index=-100,
        chunk_size=1024,
        **kwargs
    ):
        bs = labels.shape[0]
        total_size = kwargs.get("total_size", None)
        if total_size:
            chunk_size = self.calculate_chunk_size(bs, total_size)
            print_rank_0(f"[INFO] Batch size={bs}, chunk size={chunk_size}")
        labels = F.pad(labels, (0, 1), value=ignore_index)
        # Shift labels to match the input sequence for next-token prediction.
        shift_labels = labels[..., 1:].contiguous()

        # Create a mask to identify valid tokens (typically > -1 means non-special tokens)
        loss_mask = shift_labels > -1

        # Retrieve loss_type arguments to determine loss reduction behavior.
        if self.loss_type == "per_sample_loss":
            # Compute per-sample loss: alpha scales each sample by total valid tokens in the batch.
            alpha = loss_mask.sum(1) * loss_mask.shape[0]  # shape: [batch_size]
            reduction = "none"  # Keep per-token losses for sample-wise aggregation.
        elif self.loss_type == "per_token_loss":
            # Use raw sum loss without normalization here;
            avg_per_step_token_num = kwargs.get(AVG_PER_STEP_TOKEN_NUM, None)
            if avg_per_step_token_num is None:
                raise KeyError(f"per_token_loss must use PrefetchGradAccDataLoader")
            torch.distributed.all_reduce(avg_per_step_token_num, op=torch.distributed.ReduceOp.AVG)
            alpha = avg_per_step_token_num
            reduction = "sum"
        elif self.loss_type == "token_loss":
            alpha = loss_mask.sum()
            torch.distributed.all_reduce(alpha, op=torch.distributed.ReduceOp.AVG)
            reduction = "none"
        elif self.loss_type == "square_loss":
            loss_weight = (labels != -100).sum(dim=-1).float()
            loss_weight = 1 / loss_weight.sqrt()
            loss_weight = torch.where(labels != -100, loss_weight.unsqueeze(1), 0.0)
            shift_weights = loss_weight[..., 1:].contiguous().view(-1)
            shift_weight_sum = shift_weights.sum()
            torch.distributed.all_reduce(shift_weight_sum, op=torch.distributed.ReduceOp.AVG)
            alpha = shift_weight_sum / shift_weights
            reduction = "none"
        elif self.loss_type == "default":
            # Default: normalize loss by total number of valid tokens in the batch.
            alpha = loss_mask.sum() # scalar
            reduction = "sum"
        else:
            raise NotImplementedError(f"{self.loss_type} is not implemented!")

        if mpu.get_context_parallel_world_size() > 1:
            shift_labels = split_forward_gather_backward_with_cp(shift_labels, dim=-1)
            
            if self.loss_type == "square_loss":
                alpha = split_forward_gather_backward_with_cp(alpha.view(bs, -1), chunk_size, dim=1).view(-1)

        if chunk_size:
            # Split shifted labels into chunks along the sequence dimension for memory-efficient processing.
            chunk_labels = torch.split(shift_labels, chunk_size, dim=1)
            
            if self.loss_type == "square_loss":
                alpha = torch.split(alpha.view(bs, -1), chunk_size, dim=1)  

            # Prepare keyword arguments for each chunk to be passed to the chunked loss function.
            loss_ctx_kwargs = [
                {
                    "shift_labels": chunk_labels[i],
                    "ignore_index": ignore_index,
                    "reduction": reduction,
                    "alpha": alpha[i].view(-1) if isinstance(alpha, (list, tuple)) else alpha,
                }
                for i in range(len(chunk_labels))
            ]

            # Return a closure that computes the chunked language modeling loss using the prepared config.
            def loss_ctx(hidden_states, head_weight, head_bias):
                return chunk_loss(
                    hidden_states,
                    head_weight,
                    head_bias,
                    loss_forward=calculate_lm_loss,
                    loss_kwargs_chunks=loss_ctx_kwargs,
                    chunk_size=chunk_size
                )
        
        else:
            def loss_ctx(logits):
                logits = logits.view(-1, logits.shape[-1])
                labels = shift_labels.view(-1)
                return fixed_cross_entropy(
                    logits, labels,
                    alpha=alpha,
                    reduction=reduction
                )

        return loss_ctx, loss_mask

    def _set_loss_cfg(self, args):
        # Retrieve loss configuration from model.json if available
        loss_cfg = getattr(args.mm.model, "loss_cfg", None)
        # loss_cfg param: compute_mode, chunk_size, router_aux_loss_coef
        # compute_mode: default, chunk(use chunk loss)
        # chunk_size: valid when compute mode is set to chunk (default 1024)
        # router_aux_loss_coef: float (use for moe model, default 0.0)
        self.loss_compute_mode = "default"
        self.loss_chunk_size = 1024
        self.router_aux_loss_coef = 0.0
        self.loss_type = "default"
        if loss_cfg is not None:
            self.loss_compute_mode = getattr(loss_cfg, "compute_mode", "default")
            self.loss_type = getattr(loss_cfg, "loss_type", "default")
            if self.loss_compute_mode == "default":
                pass
            elif self.loss_compute_mode == "chunk":
                self.loss_chunk_size = getattr(loss_cfg, "chunk_size", 1024)
            elif self.loss_compute_mode == "dynamic_chunk":
                self.loss_chunk_size = getattr(loss_cfg, "chunk_size", 4096)
            else:
                raise NotImplementedError(f"Unrecognized loss_compute_mode: {self.loss_compute_mode}.")
            
            if self.loss_type not in ["default", "per_sample_loss", "per_token_loss", "token_loss", "square_loss"]:
                raise NotImplementedError(f"Not implemented loss_type: {self.loss_type}.")
            
            self.router_aux_loss_coef = getattr(loss_cfg, "router_aux_loss_coef", 0.0)