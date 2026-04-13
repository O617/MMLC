"""SyntheticDataLoader — random variable-length batches for scheduler testing.

Bypasses the real dataset entirely so the Scheduler and HybridScheduledDataLoader
can be stress-tested without real training data.

Each batch matches the dict structure that data_collator produces:

  Text-only mode (SYNTHETIC_WITH_VISION unset or False):
    - input_ids:      [B, max_len]  random token IDs
    - labels:         [B, max_len]  same as input_ids, -100 for padding
    - attention_mask: [B, max_len]  bool

  Vision mode (SYNTHETIC_WITH_VISION=true):
    - input_ids:      [B, max_len]  img_context_token_id × vision_tokens, then random text tokens
    - labels:         [B, max_len]  -100 for image positions and padding, random IDs for text
    - attention_mask: [B, max_len]  bool
    - pixel_values:   [total_tiles, 3, tile_hw, tile_hw]  all ones
    - image_flags:    [total_tiles]  all ones (int64)

Vision token layout per sample (length L):
    vision_tokens = floor(L × vision_ratio / tile_tokens) × tile_tokens
    num_tiles     = vision_tokens // tile_tokens
    text_tokens   = L - vision_tokens

    input_ids:  [img_ctx × vision_tokens | random_text × text_tokens | pad × (max_len - L)]
    labels:     [  -100  × vision_tokens | random_text × text_tokens | -100 × (max_len - L)]

Requirements / constraints
--------------------------
- vision_tokens must be divisible by tile_tokens (256 for InternVL3).
- The Scheduler reads image_flags with torch.sum(); if no sample has vision tokens
  all_tiles==0 triggers division by zero on line 555.  When with_vision=True and
  min_seq_len × vision_ratio < tile_tokens we log a warning and force the first
  sample to carry at least one tile.
- At least one valid label (> -1) is guaranteed per batch; a safety guard overwrites
  label[0, -1] = 1 if the entire batch is somehow all -100.

Length distributions (SYNTHETIC_LENGTH_DIST)
--------------------------------------------
  uniform      Uniform[min, max].  Default.
  normal       Gaussian centered at (min+max)/2, std = (max-min)/6, clamped.
  skewed       Right-skewed: many short seqs, moderate long tail.
               Implemented as L = min + (max-min) × U^power, U~Uniform, power=2.5.
  extreme_skew Very right-skewed: most seqs near min, very few near max.
               Same transform with power=5.0.
  burst        Occasionally inserts one ultra-long "burst" sequence.
               With probability SYNTHETIC_BURST_PROB (default 0.1), one sample
               gets SYNTHETIC_BURST_LEN tokens (default 128k).  The remaining
               B-1 samples share the leftover budget at similar moderate lengths.
               Total tokens per step is capped at cluster_size × token_budget_per_gpu.
               Steps without a burst delegate to a configurable fallback sampler.

Activation (env vars)
---------------------
  SYNTHETIC_DATA=True                enable synthetic mode (required)
  SYNTHETIC_MIN_LEN=512              minimum sequence length (default 512)
  SYNTHETIC_MAX_LEN=8192             maximum sequence length (default 8192)
  SYNTHETIC_NUM_BATCHES=1000         epoch size in batches (default 1000)
  SYNTHETIC_VOCAB_SIZE=151936        token vocabulary size (default 151936)
  SYNTHETIC_SEED=0                   base RNG seed (default 0)
  SYNTHETIC_LENGTH_DIST=uniform      length distribution: uniform|normal|skewed|extreme_skew|burst
  SYNTHETIC_WITH_VISION=false        include pixel_values / image_flags (default false)
  SYNTHETIC_VISION_RATIO=0.8         fraction of tokens that are vision tokens (default 0.8)
  SYNTHETIC_TILE_TOKENS=256          tokens per image tile — alignment unit (default 256)
  SYNTHETIC_TILE_HW=448              pixel height/width of each tile (default 448)
  SYNTHETIC_IMG_TOKEN_ID=151667      token ID used for image positions (default 151667)

  --- burst-specific env vars (only used when SYNTHETIC_LENGTH_DIST=burst) ---
  SYNTHETIC_BURST_PROB=0.1           probability per step of a burst sequence (default 0.1)
  SYNTHETIC_BURST_LEN=131072         token count for the burst sequence (default 128 × 1024)
  SYNTHETIC_CLUSTER_SIZE=64          total GPUs; budget = cluster_size × token_budget_per_gpu
  SYNTHETIC_TOKEN_BUDGET_PER_GPU=8192 per-GPU token budget (default 8192)
  SYNTHETIC_BURST_NOISE=0.1          ±fractional noise on similar-length seqs (default 0.1)
  SYNTHETIC_BURST_FALLBACK=uniform   fallback dist for non-burst steps (default uniform)

Seeding strategy
----------------
Each batch is seeded with (base_seed + step_index) so all ranks independently
produce identical batches for the same step (required for hybrid mode).
"""

from __future__ import annotations

import math
import os
import logging
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Length samplers
# ---------------------------------------------------------------------------

class LengthSampler:
    """Abstract base class for per-sample sequence length generation.

    Subclasses implement :meth:`sample` to draw ``batch_size`` integers in
    ``[min_len, max_len]`` using the supplied :class:`torch.Generator` for
    reproducibility.

    Parameters
    ----------
    min_len, max_len:
        Inclusive bounds on sequence length.
    """

    def __init__(self, min_len: int, max_len: int) -> None:
        if min_len > max_len:
            raise ValueError(f"min_len ({min_len}) > max_len ({max_len})")
        self.min_len = min_len
        self.max_len = max_len

    def sample(self, batch_size: int, rng: torch.Generator) -> list[int]:
        """Return a list of ``batch_size`` integer lengths in [min_len, max_len]."""
        raise NotImplementedError

    # Shared utility ---------------------------------------------------------

    def _to_int_lengths(self, t: torch.Tensor) -> list[int]:
        """Clamp a float tensor to [min_len, max_len] and convert to int list."""
        return t.clamp(self.min_len, self.max_len).round().long().tolist()


class UniformLengthSampler(LengthSampler):
    """Uniform distribution over [min_len, max_len].

    Every length value is equally likely.  This is the default and produces the
    most balanced coverage of the full range.
    """

    def sample(self, batch_size: int, rng: torch.Generator) -> list[int]:
        return (
            torch.randint(self.min_len, self.max_len + 1, (batch_size,), generator=rng)
            .tolist()
        )


class NormalLengthSampler(LengthSampler):
    """Gaussian distribution centered at mid-range.

    Models datasets where sequence lengths cluster symmetrically around a
    typical length, with relatively few very short or very long sequences.

    Parameters
    ----------
    std_factor:
        Standard deviation as a fraction of the full range ``(max-min)``.
        Default ``1/6`` places ±3σ at the range boundaries, so clipping is
        rare but possible for extreme draws.
    """

    def __init__(self, min_len: int, max_len: int, std_factor: float = 1 / 6) -> None:
        super().__init__(min_len, max_len)
        self.mean = (min_len + max_len) / 2.0
        self.std = (max_len - min_len) * std_factor

    def sample(self, batch_size: int, rng: torch.Generator) -> list[int]:
        samples = torch.normal(
            mean=self.mean,
            std=self.std,
            size=(batch_size,),
            generator=rng,
        )
        return self._to_int_lengths(samples)


class SkewedLengthSampler(LengthSampler):
    """Right-skewed distribution: many short sequences, moderate long tail.

    Mimics realistic VLM training datasets where most samples are short but a
    meaningful fraction is long.  Implemented via a power transform of a
    Uniform draw:

        L = min + (max - min) × U^power,   U ~ Uniform(0, 1)

    With ``power > 1`` the distribution concentrates near ``min_len``.
    The default ``power=2.5`` gives a moderate right skew: the median maps to
    roughly ``min + (max-min) × 0.5^2.5 ≈ min + 0.18 × (max-min)``.

    Parameters
    ----------
    power:
        Exponent for the power transform.  Larger values produce a more
        extreme concentration near ``min_len``.  Must be ≥ 1.
    """

    def __init__(self, min_len: int, max_len: int, power: float = 2.5) -> None:
        super().__init__(min_len, max_len)
        if power < 1:
            raise ValueError(f"power must be >= 1, got {power}")
        self.power = power

    def sample(self, batch_size: int, rng: torch.Generator) -> list[int]:
        u = torch.rand(batch_size, generator=rng)           # U ~ Uniform(0, 1)
        frac = u.pow(self.power)                            # concentrate near 0
        samples = self.min_len + (self.max_len - self.min_len) * frac
        return self._to_int_lengths(samples)


class ExtremeSkewedLengthSampler(SkewedLengthSampler):
    """Extremely right-skewed: almost all sequences near min_len, very few long.

    Uses the same power-transform as :class:`SkewedLengthSampler` but with a
    much larger default exponent (``power=5.0``).  The median maps to roughly
    ``min + (max-min) × 0.5^5 ≈ min + 0.03 × (max-min)``.

    Useful for stress-testing the Scheduler's handling of workloads dominated
    by short sequences with occasional very long outliers.
    """

    def __init__(self, min_len: int, max_len: int, power: float = 5.0) -> None:
        super().__init__(min_len, max_len, power=power)


class BurstLengthSampler(LengthSampler):
    """Occasionally injects one ultra-long "burst" sequence per batch.

    With probability ``burst_prob`` this sampler generates a batch where:

    * One sequence has exactly ``burst_len`` tokens (the "burst").
    * The remaining ``B-1`` sequences receive similar moderate lengths that
      together fit within the remaining token budget:

          target = (cluster_size × token_budget_per_gpu - burst_len) / (B - 1)

      Each of the B-1 lengths is then jittered by ±``burst_noise`` × target
      and clamped to ``[min_len, max_len]``.

    Without a burst (probability ``1 - burst_prob``), the call is forwarded to
    ``fallback_sampler`` unchanged.

    **Total-token budget guarantee**: in burst mode the sum of all lengths is
    at most ``cluster_size × token_budget_per_gpu``.  In fallback mode the
    budget is not re-enforced — configure ``min_len`` / ``max_len`` of the
    fallback sampler accordingly.

    Parameters
    ----------
    min_len, max_len:
        Inclusive length bounds for *normal* (non-burst) sequences.
    burst_prob:
        Probability per step that a burst sequence is inserted (default 0.1).
    burst_len:
        Token count of the burst sequence (default 128 × 1024 = 131 072).
        May exceed ``max_len``; the budget constraint is the only upper bound.
    cluster_size:
        Total number of GPUs.  Token budget per step =
        ``cluster_size × token_budget_per_gpu``.
    token_budget_per_gpu:
        Per-GPU token budget (default 8 192).
    burst_noise:
        Fractional noise ± applied to the "similar length" target for the
        non-burst sequences in a burst step (default 0.1 = ±10 %).
    fallback_sampler:
        :class:`LengthSampler` used for non-burst steps.  Defaults to
        :class:`UniformLengthSampler` over ``[min_len, max_len]``.
    """

    def __init__(
        self,
        min_len: int,
        max_len: int,
        burst_prob: float = 0.1,
        burst_len: int = 128 * 1024,
        cluster_size: int = 64,
        token_budget_per_gpu: int = 8192,
        burst_noise: float = 0.1,
        fallback_sampler: LengthSampler | None = None,
    ) -> None:
        # Parent stores the full possible range [min_len, burst_len]
        super().__init__(min_len, max(max_len, burst_len))
        self._normal_max = max_len
        self.burst_prob = burst_prob
        self.burst_len = burst_len
        self.max_total_tokens = cluster_size * token_budget_per_gpu
        self.burst_noise = burst_noise
        self.fallback_sampler = fallback_sampler or UniformLengthSampler(min_len, max_len)

        if burst_len > self.max_total_tokens:
            raise ValueError(
                f"burst_len ({burst_len:,}) exceeds the total token budget "
                f"({cluster_size} GPUs × {token_budget_per_gpu} = "
                f"{self.max_total_tokens:,}). "
                f"Reduce burst_len or increase cluster_size / token_budget_per_gpu."
            )

    def sample(self, batch_size: int, rng: torch.Generator) -> list[int]:
        # Draw burst trigger — a single uniform sample consumed before any
        # branching so the rng state advances identically on every rank.
        trigger = torch.rand(1, generator=rng).item()

        if trigger >= self.burst_prob:
            return self.fallback_sampler.sample(batch_size, rng)

        # ── Burst mode ────────────────────────────────────────────────────
        if batch_size == 1:
            return [self.burst_len]

        n_rest = batch_size - 1
        remaining_budget = self.max_total_tokens - self.burst_len

        # Target length for each of the B-1 companion sequences
        if remaining_budget <= n_rest * self.min_len:
            # Edge case: budget too tight even for min_len — give everyone min_len
            target = float(self.min_len)
        else:
            target = remaining_budget / n_rest
            target = min(target, float(self._normal_max))

        # Jitter: ±burst_noise × target, then clamp to [min_len, normal_max]
        noise = (torch.rand(n_rest, generator=rng) * 2.0 - 1.0) * (target * self.burst_noise)
        rest_lengths = (
            (torch.full((n_rest,), target) + noise)
            .clamp(self.min_len, self._normal_max)
            .round()
            .long()
            .tolist()
        )

        # Insert the burst sequence at a random position within the batch
        burst_pos = int(torch.randint(0, batch_size, (1,), generator=rng).item())
        lengths = list(rest_lengths)
        lengths.insert(burst_pos, self.burst_len)

        return lengths


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SAMPLER_REGISTRY: dict[str, type[LengthSampler]] = {
    "uniform":      UniformLengthSampler,
    "normal":       NormalLengthSampler,
    "skewed":       SkewedLengthSampler,
    "extreme_skew": ExtremeSkewedLengthSampler,
    # "burst" is handled specially in make_length_sampler / build_synthetic_dataloader
    # because it requires extra keyword arguments beyond (min_len, max_len).
}


def make_length_sampler(
    dist: str,
    min_len: int,
    max_len: int,
    **burst_kwargs,
) -> LengthSampler:
    """Instantiate a :class:`LengthSampler` by name.

    Parameters
    ----------
    dist:
        Distribution name: ``"uniform"``, ``"normal"``, ``"skewed"``,
        ``"extreme_skew"``, or ``"burst"``.
    min_len, max_len:
        Inclusive length bounds passed to the sampler.
    **burst_kwargs:
        Extra keyword arguments forwarded to :class:`BurstLengthSampler` when
        ``dist == "burst"`` (e.g. ``burst_prob``, ``burst_len``,
        ``cluster_size``, ``token_budget_per_gpu``, ``burst_noise``,
        ``fallback_sampler``).

    Raises
    ------
    ValueError
        If ``dist`` is not a recognised distribution name.
    """
    dist = dist.lower().strip()
    if dist == "burst":
        return BurstLengthSampler(min_len, max_len, **burst_kwargs)
    if dist not in _SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown length distribution {dist!r}. "
            f"Choose from: {sorted(_SAMPLER_REGISTRY) + ['burst']}"
        )
    return _SAMPLER_REGISTRY[dist](min_len, max_len)


# ---------------------------------------------------------------------------
# SyntheticDataLoader
# ---------------------------------------------------------------------------

class SyntheticDataLoader:
    """Iterable that yields deterministic random variable-length batches.

    Parameters
    ----------
    batch_size:
        Sequences per batch.  In hybrid parallel mode pass the *inflated* MBS
        (micro_batch_size × num_groups) so the Scheduler has enough samples.
    min_seq_len / max_seq_len:
        Inclusive range for per-sample sequence lengths.
    vocab_size:
        Token vocabulary size; random IDs are sampled from [1, vocab_size).
    num_batches:
        Epoch length.  build_iterations wraps this in a cyclic iterator.
    seed:
        Base RNG seed; actual seed for step i is (seed + i).
    length_sampler:
        A :class:`LengthSampler` instance controlling the length distribution.
        If *None*, a :class:`UniformLengthSampler` over [min_seq_len, max_seq_len]
        is created automatically.
    with_vision:
        If True, include pixel_values and image_flags in each batch.
    vision_token_ratio:
        Fraction of each sequence's tokens that are vision tokens.
        Rounded down to the nearest multiple of tile_tokens.
    tile_tokens:
        Number of tokens produced by one image tile (256 for InternVL3).
    tile_hw:
        Pixel height and width of each tile (448 for InternVL3).
        pixel_values shape: [total_tiles, 3, tile_hw, tile_hw].
    img_context_token_id:
        Token ID placed in input_ids at every vision token position.
        Must match the value in the model's JSON config (151667 for InternVL3).
    """

    def __init__(
        self,
        batch_size: int,
        min_seq_len: int = 512,
        max_seq_len: int = 8192,
        vocab_size: int = 151936,
        num_batches: int = 1000,
        seed: int = 0,
        length_sampler: LengthSampler | None = None,
        with_vision: bool = False,
        vision_token_ratio: float = 0.8,
        tile_tokens: int = 256,
        tile_hw: int = 448,
        img_context_token_id: int = 151667,
    ) -> None:
        if not (0.0 < vision_token_ratio < 1.0):
            raise ValueError(
                f"vision_token_ratio must be in (0, 1), got {vision_token_ratio}"
            )

        self.batch_size = batch_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_batches = num_batches
        self.seed = seed
        self.length_sampler = length_sampler or UniformLengthSampler(min_seq_len, max_seq_len)
        self.with_vision = with_vision
        self.vision_token_ratio = vision_token_ratio
        self.tile_tokens = tile_tokens
        self.tile_hw = tile_hw
        self.img_context_token_id = img_context_token_id
        self._count = 0

        if with_vision:
            min_vision = int(min_seq_len * vision_token_ratio)
            if min_vision < tile_tokens:
                logger.warning(
                    "[SyntheticData] min_seq_len=%d × vision_ratio=%.2f = %d < tile_tokens=%d. "
                    "Some short sequences may have 0 vision tokens; the first sample in such "
                    "batches will be forced to carry at least one tile to avoid Scheduler "
                    "division-by-zero.",
                    min_seq_len, vision_token_ratio, min_vision, tile_tokens,
                )

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> "SyntheticDataLoader":
        self._count = 0
        return self

    def __next__(self) -> dict:
        if self._count >= self.num_batches:
            raise StopIteration
        batch = self._make_batch(self._count)
        self._count += 1
        return batch

    def __len__(self) -> int:
        return self.num_batches

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vision_tokens_for_len(self, length: int) -> int:
        """Round vision token count DOWN to the nearest multiple of tile_tokens."""
        raw = int(length * self.vision_token_ratio)
        return (raw // self.tile_tokens) * self.tile_tokens

    def _make_batch(self, step: int) -> dict:
        """Generate a deterministic random batch for the given step index.

        The step index is folded into the seed so every rank independently
        produces the same batch for the same step (cross-rank parity).
        """
        rng = torch.Generator()
        rng.manual_seed(self.seed + step)

        B = self.batch_size

        # 1. Sample per-sequence lengths via the configured distribution
        lengths = self.length_sampler.sample(B, rng)
        max_len = max(lengths)

        # 2. Compute per-sample vision token counts (may be 0 for short seqs)
        if self.with_vision:
            vis_tokens = [self._vision_tokens_for_len(l) for l in lengths]
            # Guard: all-zero tiles → Scheduler line 555 divides by zero.
            # Force the first sample to carry at least one tile.
            if sum(vis_tokens) == 0:
                vis_tokens[0] = self.tile_tokens
                if lengths[0] < self.tile_tokens:
                    lengths[0] = self.tile_tokens
                    max_len = max(max_len, self.tile_tokens)
        else:
            vis_tokens = [0] * B

        # 3. Build per-sample tensors
        input_ids = torch.zeros(B, max_len, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i in range(B):
            l = lengths[i]
            vt = vis_tokens[i]
            tt = l - vt                 # text token count

            # Vision positions: filled with img_context_token_id, labels stay -100
            if vt > 0:
                input_ids[i, :vt] = self.img_context_token_id

            # Text positions: random token IDs (avoid 0=pad and img_context_token_id)
            if tt > 0:
                text_toks = torch.randint(1, self.vocab_size, (tt,), generator=rng)
                text_toks[text_toks == self.img_context_token_id] = 1
                input_ids[i, vt:l] = text_toks
                labels[i, vt:l] = text_toks

            attention_mask[i, :l] = True

        # 4. Safety guard: guarantee at least one valid label in the whole batch
        if (labels > -1).sum() == 0:
            l0, vt0 = lengths[0], vis_tokens[0]
            if l0 > vt0:
                labels[0, l0 - 1] = 1
                input_ids[0, l0 - 1] = 1
            elif l0 < max_len:
                input_ids[0, l0] = 1
                labels[0, l0] = 1
                attention_mask[0, l0] = True

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # 5. Build pixel_values and image_flags (vision mode only)
        if self.with_vision:
            num_tiles = [vt // self.tile_tokens for vt in vis_tokens]
            total_tiles = sum(num_tiles)
            batch["pixel_values"] = torch.ones(
                total_tiles, 3, self.tile_hw, self.tile_hw, dtype=torch.float32
            )
            batch["image_flags"] = torch.ones(total_tiles, dtype=torch.long)

        return batch


# ---------------------------------------------------------------------------
# Builder (reads from env vars)
# ---------------------------------------------------------------------------

def build_synthetic_dataloader(batch_size: int) -> SyntheticDataLoader:
    """Construct a :class:`SyntheticDataLoader` from environment variables.

    Call this *after* any MBS inflation so ``batch_size`` already reflects the
    total samples the Scheduler expects per step.
    """
    def _env(key: str, default):
        return os.environ.get(key, default)

    min_len = int(_env("SYNTHETIC_MIN_LEN", 512))
    max_len = int(_env("SYNTHETIC_MAX_LEN", 8192))
    dist    = _env("SYNTHETIC_LENGTH_DIST", "uniform").lower().strip()

    if dist == "burst":
        fallback_dist = _env("SYNTHETIC_BURST_FALLBACK", "uniform")
        fallback_sampler = make_length_sampler(fallback_dist, min_len, max_len)
        length_sampler = make_length_sampler(
            "burst", min_len, max_len,
            burst_prob=float(_env("SYNTHETIC_BURST_PROB", 0.1)),
            burst_len=int(_env("SYNTHETIC_BURST_LEN", 128 * 1024)),
            cluster_size=int(_env("SYNTHETIC_CLUSTER_SIZE", 64)),
            token_budget_per_gpu=int(_env("SYNTHETIC_TOKEN_BUDGET_PER_GPU", 8192)),
            burst_noise=float(_env("SYNTHETIC_BURST_NOISE", 0.1)),
            fallback_sampler=fallback_sampler,
        )
    else:
        length_sampler = make_length_sampler(dist, min_len, max_len)

    with_vision = _env("SYNTHETIC_WITH_VISION", "false").lower() == "true"

    return SyntheticDataLoader(
        batch_size=batch_size,
        min_seq_len=min_len,
        max_seq_len=max_len,
        vocab_size=int(_env("SYNTHETIC_VOCAB_SIZE", 151936)),
        num_batches=int(_env("SYNTHETIC_NUM_BATCHES", 1000)),
        seed=int(_env("SYNTHETIC_SEED", 0)),
        length_sampler=length_sampler,
        with_vision=with_vision,
        vision_token_ratio=float(_env("SYNTHETIC_VISION_RATIO", 0.8)),
        tile_tokens=int(_env("SYNTHETIC_TILE_TOKENS", 256)),
        tile_hw=int(_env("SYNTHETIC_TILE_HW", 448)),
        img_context_token_id=int(_env("SYNTHETIC_IMG_TOKEN_ID", 151667)),
    )
