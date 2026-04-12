"""HybridScheduledDataLoader

A main-process iterator wrapper that integrates the Scheduler (and
DoubleBufferedScheduler) directly into the data pipeline.  The training
loop consumes this object exactly like a standard DataLoader — no
hybrid-parallel awareness is required outside this file.

Responsibilities
----------------
1. Load raw batches from the underlying DataLoader.
2. Move tensors to device / dtype (replaces pretrain_vlm._load_raw_batch).
3. Handle the video key rename (pixel_values_videos → pixel_values).
4. Optionally apply EncoderBalanceComm (replaces _apply_encoder_balance).
5. Run Scheduler.next_batch() which:
      a. Computes the CP-group assignment (naive or dynamic BFD).
      b. Calls the pixel_loader to load only the per-rank images (two-phase).
      c. Returns the slice of the batch for this rank's CP group.
6. For DoubleBufferedScheduler: manage the double-buffer prefetch state so
   scheduling + image I/O overlap with the GPU forward/backward pass.

Usage (in train_valid_test_datasets_provider)
---------------------------------------------
    from hybrid_dataloader import HybridScheduledDataLoader

    base_loader = build_mm_dataloader(...)
    hybrid_loader = HybridScheduledDataLoader(
        base_loader=base_loader,
        scheduler=data_scheduler,
        float_dtype=args.params_dtype,
        encoder_dp_balance=args.encoder_dp_balance,
    )
    # Pass hybrid_loader wherever a regular dataloader is expected.
    # get_batch() in pretrain_vlm.py is then just: return next(data_iterator)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
from megatron.core import mpu
from megatron.training import get_args

from scheduler import DoubleBufferedScheduler, Scheduler
from mindspeed_mm.utils.utils import EncoderBalanceComm


# ---------------------------------------------------------------------------
# Helpers (previously private functions in pretrain_vlm.py)
# ---------------------------------------------------------------------------

def _move_to_device(batch: Dict[str, Any], float_dtype: str) -> None:
    """Cast and move all tensors in batch to current device in-place."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            batch[k] = v.to(device=torch.cuda.current_device(), dtype=dtype)
        elif isinstance(v, list) and v and all(isinstance(t, torch.Tensor) for t in v):
            batch[k] = [
                t.to(device=torch.cuda.current_device(),
                     dtype=float_dtype if torch.is_floating_point(t) else None)
                for t in v
            ]


def _normalize_video_keys(batch: Dict[str, Any]) -> None:
    """Rename video-specific keys to the canonical pixel_values key."""
    if 'pixel_values_videos' in batch and 'video_grid_thw' in batch:
        batch['pixel_values'] = batch.pop('pixel_values_videos')
        batch['image_grid_thw'] = batch.pop('video_grid_thw')


def _apply_encoder_balance(batch: Dict[str, Any], is_vit_last_stage: bool,
                            encoder_dp_balance: bool) -> None:
    """Apply EncoderBalanceComm if this stage owns the ViT output."""
    if (mpu.is_pipeline_first_stage() or is_vit_last_stage) and encoder_dp_balance:
        batch['pixel_values'], batch['tranfer'] = EncoderBalanceComm.apply(
            batch['pixel_values'],
            mpu.get_data_parallel_group())
    else:
        batch['tranfer'] = None


def _diag_raw_batch(batch: Dict[str, Any]) -> None:
    """Diagnostic print for the raw batch (before Scheduler slicing)."""
    rank = torch.distributed.get_rank()
    labels = batch.get('labels')
    if labels is not None:
        shape = tuple(labels.shape)
        flat = labels.flatten()
        nonpad = flat[flat > -1]
        valid = int(nonpad.numel())
        label_sum = int(nonpad.sum())
        first20 = nonpad[:20].tolist()
    else:
        shape, valid, label_sum, first20 = None, 0, 0, []
    print(f"[DIAG-RAW] rank={rank} labels_shape={shape} "
          f"valid_tokens={valid} label_sum={label_sum} "
          f"first20_labels={first20}")


# ---------------------------------------------------------------------------
# HybridScheduledDataLoader
# ---------------------------------------------------------------------------

class HybridScheduledDataLoader:
    """Main-process iterator that fuses DataLoader + Scheduler.

    All distributed collectives inside Scheduler (barrier, create_group, etc.)
    are safe here because every rank calls ``__next__`` in lock-step, which is
    guaranteed by Megatron's training loop.

    Parameters
    ----------
    base_loader:
        The underlying DataLoader (already configured with _SingleRankGroup
        and inflated MBS for hybrid parallel).
    scheduler:
        A ``Scheduler`` or ``DoubleBufferedScheduler`` instance.
    float_dtype:
        Target dtype for floating-point tensors (e.g. ``torch.bfloat16``).
    encoder_dp_balance:
        Whether to apply EncoderBalanceComm after loading.
    """

    def __init__(
        self,
        base_loader,
        scheduler: Scheduler,
        float_dtype: str,
        encoder_dp_balance: bool = False,
    ) -> None:
        self.base_loader = base_loader
        self.scheduler = scheduler
        self.float_dtype = float_dtype
        self.encoder_dp_balance = encoder_dp_balance

        self._base_iter = None
        # Double-buffer prefetch state (used only with DoubleBufferedScheduler)
        self._prefetch_raw_batch: Optional[Dict[str, Any]] = None
        self._prefetch_started: bool = False

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> "HybridScheduledDataLoader":
        self._base_iter = iter(self.base_loader)
        self._prefetch_raw_batch = None
        self._prefetch_started = False
        return self

    def __next__(self) -> Dict[str, Any]:
        # is_vit_last_stage is not known here; callers that need it should
        # pass it via set_vit_last_stage() before starting iteration, or
        # we fall back to False (safe: EncoderBalanceComm is a no-op then).
        return self._next_batch(is_vit_last_stage=getattr(self, '_vit_last_stage', False))

    def __len__(self) -> int:
        return len(self.base_loader)

    def set_vit_last_stage(self, value: bool) -> None:
        """Set whether the current rank is the last ViT pipeline stage.

        Call this once from forward_step (or model_provider) before training
        begins if encoder_dp_balance is enabled.
        """
        self._vit_last_stage = value

    # ------------------------------------------------------------------
    # Internal batch-fetch logic
    # ------------------------------------------------------------------

    def _load_one(self) -> Dict[str, Any]:
        """Pull one raw batch from the underlying iterator and prepare it."""
        batch = next(self._base_iter)           # may raise StopIteration
        _move_to_device(batch, self.float_dtype)
        _normalize_video_keys(batch)
        _diag_raw_batch(batch)
        return batch

    def _prepare(self, batch: Dict[str, Any], is_vit_last_stage: bool) -> Dict[str, Any]:
        """Apply encoder balance and run Scheduler on a raw batch."""
        _apply_encoder_balance(batch, is_vit_last_stage, self.encoder_dp_balance)
        return self.scheduler.next_batch(batch)

    def _next_batch(self, is_vit_last_stage: bool) -> Dict[str, Any]:
        is_async = isinstance(self.scheduler, DoubleBufferedScheduler)

        if is_async:
            return self._next_async(is_vit_last_stage)
        else:
            return self._next_sync(is_vit_last_stage)

    # -- Synchronous path (plain Scheduler) ----------------------------

    def _next_sync(self, is_vit_last_stage: bool) -> Dict[str, Any]:
        raw = self._load_one()
        return self._prepare(raw, is_vit_last_stage)

    # -- Async path (DoubleBufferedScheduler) --------------------------

    def _next_async(self, is_vit_last_stage: bool) -> Dict[str, Any]:
        """Double-buffered path.

        Timeline
        --------
        step N:  _next_async returns the already-prefetched result for step N
                 and immediately kicks off prefetch for step N+1.
        step N+1: same — the GPU is running forward/backward on step N while
                  the background thread computes step N+1's schedule + images.
        """
        sched: DoubleBufferedScheduler = self.scheduler  # type: ignore[assignment]

        if not sched.is_warmup() and self._prefetch_started:
            # ── Fast path: consume prefetched result ──────────────────
            batch = sched.swap_and_get_data()
            # EncoderBalanceComm was applied to the raw batch before prefetch
            # and pixel_values were already filtered by get_data(); just
            # propagate the 'tranfer' key from the saved raw batch.
            batch['tranfer'] = (
                self._prefetch_raw_batch.get('tranfer')
                if self._prefetch_raw_batch else None
            )

            # Kick off prefetch for the NEXT step
            self._start_prefetch(is_vit_last_stage)
            return batch

        # ── Slow path: synchronous (warmup or first call) ─────────────
        raw = self._load_one()
        _apply_encoder_balance(raw, is_vit_last_stage, self.encoder_dp_balance)
        batch = sched.next_batch(raw)

        # After warmup completes, kick off the very first prefetch
        if not sched.is_warmup() and not self._prefetch_started:
            self._start_prefetch(is_vit_last_stage)

        return batch

    def _start_prefetch(self, is_vit_last_stage: bool) -> None:
        """Load the next raw batch and hand it to the background scheduler."""
        sched: DoubleBufferedScheduler = self.scheduler  # type: ignore[assignment]
        try:
            next_raw = self._load_one()
            _apply_encoder_balance(next_raw, is_vit_last_stage, self.encoder_dp_balance)
            sched.start_prefetch(next_raw)
            self._prefetch_raw_batch = next_raw
            self._prefetch_started = True
        except StopIteration:
            # Epoch boundary — no more data to prefetch
            self._prefetch_started = False
            self._prefetch_raw_batch = None


# ---------------------------------------------------------------------------
# Dataset helpers (previously in pretrain_vlm.py)
# ---------------------------------------------------------------------------

def get_img_processor_from_dataset(dataset):
    """Extract img_video_processor from a dataset (handles ConcatDataset)."""
    if hasattr(dataset, 'img_video_processor'):
        return dataset.img_video_processor
    if hasattr(dataset, 'datasets') and dataset.datasets:
        return get_img_processor_from_dataset(dataset.datasets[0])
    return None


def set_skip_pixel_values(dataset, value: bool = True) -> None:
    """Recursively enable/disable lightweight (skip pixel) mode on datasets."""
    if hasattr(dataset, 'skip_pixel_values'):
        dataset.skip_pixel_values = value
    if hasattr(dataset, 'datasets'):
        for d in dataset.datasets:
            set_skip_pixel_values(d, value)


def make_pixel_loader(img_processor):
    """Return a pixel_loader callback for two-phase image loading.

    The callback is registered with ``scheduler.set_pixel_loader()`` and
    called inside ``Scheduler.get_data()`` after the scheduling decision is
    made.  It loads pixel_values only for the samples assigned to this rank.

    Signature: pixel_loader(databatch, local_data_ids) -> (pixel_values, image_flags)
    """
    def pixel_loader(databatch, local_data_ids):
        image_paths_all = databatch.get('_image_path', [])
        image_modes_all = databatch.get('_image_mode', [])

        all_pixel_values = []
        all_flags = []

        for idx in local_data_ids:
            paths = image_paths_all[idx]        # list of path strings
            mode = image_modes_all[idx] if image_modes_all else 'single_image'

            if mode == 'single_image':
                pv = img_processor(image_path=paths[0], mode='single_image',
                                   num_image=1)['pixel_values']
            elif mode == 'multi_image':
                pv_parts = []
                num_images = len(paths)
                for p in paths:
                    cur = img_processor(image_path=p, mode='multi_image',
                                        num_image=num_images)['pixel_values']
                    pv_parts.extend(cur)
                pv = torch.stack(pv_parts)
            elif mode == 'video':
                pv = img_processor(video_path=paths[0])['pixel_values']
            else:
                raise ValueError(f"Unknown image mode: {mode}")

            all_pixel_values.append(pv)
            all_flags.extend([1] * pv.shape[0])

        pixel_values = torch.cat(all_pixel_values, dim=0) if all_pixel_values else None
        image_flags = torch.tensor(all_flags, dtype=torch.long)
        return pixel_values, image_flags

    return pixel_loader
