import torch
import psutil
import os
import gc
import time
import threading
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from functools import wraps
from megatron.core import mpu
from mindspeed.core.context_parallel import model_parallel_utils as ms_mpu
from megatron.core.packed_seq_params import PackedSeqParams
from scipy.optimize import linear_sum_assignment

def monitor_memory(func):
    """装饰器：监测函数内存使用"""
    def wrapper(*args, **kwargs):
        gc.collect()

        process = psutil.Process(os.getpid())

        mem_before = process.memory_info().rss / 1024 / 1024 / 1024 # GB

        result = func(*args, **kwargs)

        gc.collect()

        mem_after = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        
        if torch.distributed.get_rank() == 0:
            print(f"函数 {func.__name__} 内存使用：")
            print(f"  运行前: {mem_before:.2f} GB")
            print(f"  运行后: {mem_after:.2f} GB")
            print(f"  增加: {mem_after - mem_before:.2f} GB")
        
        return result
    return wrapper

def timing(func):
    """测量函数运行时间的基础装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()  # 高精度计时
        result = func(*args, **kwargs)
        end = time.perf_counter()
        if torch.distributed.get_rank() == ():
            print(f"[⏱️] {func.__name__} 运行时间: {(end - start) * 1000:.2f} ms")
        return result
    return wrapper


def pad_len_to(data_len, pad_num):
    return ((data_len + pad_num - 1) // pad_num) * pad_num


class Scheduler:
    def __init__(self, cluster_size, other_parallel_group_size, img_context_token_id, cp_window_size,
                 schedule_mode="dynamic", seq_len_chunk=8704, max_cp_degree=32,
                 cp_must_be_power_of_two=False):
        self.data_dict = None
        self.hybrid_group = None
        self.cluster_size = cluster_size
        self.other_parallel_group_size = other_parallel_group_size
        self.rank = torch.distributed.get_rank()
        self.rank_dicts = [{}] * self.other_parallel_group_size
        self.group_id = None
        self.data_len = None
        self.img_context_token_id = img_context_token_id
        self.group_pool = {}
        self.cp_window_size = int(cp_window_size) if cp_window_size is not None else 1
        self.dp_group = None
        self.dp_group_ranks = None
        self.schedule_mode = schedule_mode
        self.pixel_loader = None  # Optional callback for two-phase image loading
        # Per-rank token budget used by BFD packing and min_cp computation.
        # Coupling invariant: synthetic_dataloader's token_budget_per_gpu MUST equal
        # this value so the producer never hands the scheduler more tokens than the
        # cluster can absorb.  InternVL3 (VLMModel path) uses 9216 historically;
        # Qwen3VL (TransformersModel/HF path) uses 8192 so ceil(256k/chunk)=32 keeps
        # the largest supported burst within CP<=32.
        self.seq_len_chunk = int(seq_len_chunk)
        # Upper bound on CP degree any group may receive.  For Ulysses this must
        # be ≤ num_attention_heads of the model (e.g. 32 for Qwen3VL-4B, 16 for 2B)
        # because Ulysses splits Q heads across CP ranks.  For Ring it can be any
        # positive integer ≤ cluster size.
        self.max_cp_degree = int(max_cp_degree)
        # Constraint on CP values produced by the rank-conservation inflation
        # post-process.  When True (Ulysses path), every inflated CP must be a
        # power of 2 so that num_attention_heads divides it cleanly — this
        # restricts inflation to CP ← 2×CP doublings.  When False (Ring path,
        # the InternVL3 default), CP can be any positive integer ≤ max_cp_degree
        # so inflation grows groups by +1 in round-robin.
        self.cp_must_be_power_of_two = bool(cp_must_be_power_of_two)

    def set_pixel_loader(self, pixel_loader):
        """Set a callback for loading pixel_values after scheduling (two-phase mode).

        The callback signature is:
            pixel_loader(databatch, local_data_ids) -> (pixel_values_tensor, image_flags_tensor)

        where databatch is the raw full batch and local_data_ids is a list of
        sample indices assigned to this rank's CP group.
        """
        self.pixel_loader = pixel_loader

    def get_group_data(self, group_id):
        assert self.rank_dicts is not None, "Rank dict is not initialized" 
        return self.data_dict[group_id]
    
    def update_rank_dicts(self, group_list, ):
        assert sum(group_list) * self.other_parallel_group_size <= self.cluster_size, "The parallel size of hybrid-parallel group must be less than Cluster Size"
        # Clear stale entries from previous batches
        self.rank_dicts = [{}] * self.other_parallel_group_size
        cumsum = 0
        dp_group_ranks = []
        torch.distributed.barrier()
        for group_id, group_size in enumerate(group_list):
            group_ranks = [(i + cumsum) * self.other_parallel_group_size for i in range(group_size)]
            dp_group_ranks.append(group_ranks[0] + self.rank % self.other_parallel_group_size)
            for other_parallel_group_id in range(self.other_parallel_group_size):
                self.rank_dicts[other_parallel_group_id][str(group_id)] = [group_rank + other_parallel_group_id for group_rank in group_ranks]
            cumsum += group_size

    def get_num_groups(self):
        """Return the number of CP groups in the current batch."""
        rank_dict_local = self.rank_dicts[self.rank % self.other_parallel_group_size]
        return len(rank_dict_local)

    def get_from_group_pool(self, rank_list, allow_create=True):
        group_uid = self.get_group_uid(rank_list)
        if group_uid in self.group_pool.keys():
            return self.group_pool[group_uid]
        if not allow_create:
            # mpu.create_group() is a collective op that requires all ranks to
            # participate simultaneously.  Calling it from a background thread
            # (DoubleBufferedScheduler._prefetch_worker) would deadlock because
            # the other ranks are running forward/backward, not waiting here.
            # This means warmup did not cover all group configurations — either
            # extend warmup steps or file a bug report.
            raise RuntimeError(
                f"Group pool cache miss for ranks {rank_list} after warmup. "
                "mpu.create_group() cannot be called from a background thread. "
                "Increase warmup steps so all CP group configurations are cached."
            )
        group = mpu.create_group(ranks=rank_list, use_local_synchronization=False)
        self.group_pool[group_uid] = group
        return group

    def get_group_id(self, rank_id):
        assert self.rank_dicts is not None, "Rank dict is not initialized"
        rank_dict_local = self.rank_dicts[self.rank % self.other_parallel_group_size]
        for group_id in rank_dict_local.keys():
            if rank_id in rank_dict_local[group_id]:
                return group_id
        raise ValueError(f"Rank {rank_id} is not in any parallel group")
    
    def get_group_uid(self, rank_list):
        uid = ''
        for rank in rank_list:
            uid += f'{rank}_'
        return uid
    
    def update_parallel_group(self,):
        is_initialized = False
        self.group_id = self.get_group_id(self.rank)
        cp_size = None
        rank_dict_local = self.rank_dicts[self.rank % self.other_parallel_group_size]
        for group_id in rank_dict_local.keys():
            group = self.get_from_group_pool(rank_dict_local[str(group_id)])
            if self.rank in rank_dict_local[group_id]:
                assert is_initialized is False, "Hybrid-parallel group is initialized again in one GBS"
                self.hybrid_group = group
                cp_size = len(rank_dict_local[group_id])
                is_initialized = True
        assert is_initialized is True, f"Rank {self.rank} is not in any hybrid-parallel group"
        local_group_ranks =  rank_dict_local[str(self.group_id)]
        mpu._CONTEXT_PARALLEL_GROUP = self.hybrid_group
        mpu._CONTEXT_PARALLEL_GLOBAL_RANKS = local_group_ranks
        ms_mpu._CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = self.hybrid_group
        # mpu._DATA_PARALLEL_GROUP = self.dp_group
        # mpu._DATA_PARALLEL_GLOBAL_RANKS
        ms_mpu._CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV = [local_group_ranks[idx] for idx in range(0, cp_size, 1)]
        ms_mpu._CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV = [local_group_ranks[idx] for idx in range(0, cp_size, 1)]

    def update_group_data_id(self, data_list):
        for group_id, data in enumerate(data_list):
            self.data_dict[str(group_id)] = data

    def mock_time_estimator(self, seqlens, cp_size):
        """
        Mock function to estimate computation time for a sequence.
        Simulates O(L^2) attention complexity with CP speedup and communication overhead.
        
        Args:
            seqlen: sequence length (number of tokens)
            cp_size: context parallelism degree
        
        Returns:
            estimated time in arbitrary units (seconds)
        """
        seqlens = seqlens if type(seqlens) is list else [seqlens]
        total_time = 0.0
        for seqlen in seqlens:
            # Simulate attention computation: O(L^2) with parallelism speedup
            base_compute = (seqlen ** 2) / (cp_size * 1e7)  # Normalized compute time
            # Linear overhead from layer norm, MLP, etc.
            linear_overhead = seqlen / (cp_size * 1e5)
            # Communication overhead (ring attention P2P)
            comm_overhead = seqlen / (cp_size * 2e5)
            # Small constant overhead for kernel launch, synchronization
            constant_overhead = 0.001
            total_time = total_time + base_compute + linear_overhead + comm_overhead + constant_overhead
        return total_time


    def memory_constraint_check(self, sequence_lengths, cp_degree, memory_budget_per_rank=1.0):
        """
        Check if a group of sequences can fit within memory constraints.
        Currently always returns True as requested.
        
        Args:
            sequence_lengths: list of sequence lengths in the group
            cp_degree: context parallelism degree for this group
            memory_budget_per_rank: memory budget per rank (GB)
        
        Returns:
            bool: whether the group satisfies memory constraints
        """
        return True


    def sequence_packing_bfd(self, sequences, min_cp_degrees, capacity_factor=8192):
        """
        Stage 1: Memory-aware Sequence Packing using Best-Fit Decreasing (BFD).
        
        Groups heterogeneous sequences into atomic groups to reduce decision variables
        and avoid memory fragmentation.
        
        Args:
            sequences: list of (seq_id, seqlen) tuples
            min_cp_degrees: dict mapping seq_id to minimum required CP degree
            capacity_factor: base capacity unit for bin packing (tokens)
        
        Returns:
            groups: list of dicts with keys:
                - 'seq_ids': list of sequence indices
                - 'min_cp': minimum CP degree required
                - 'total_len': sum of sequence lengths
                - 'used_capacity': current used capacity in the bin
        """
        if not sequences:
            return []
        
        # Sort by length descending (Best-Fit Decreasing heuristic)
        sorted_seqs = sorted(sequences, key=lambda x: x[1], reverse=True)
        
        groups = []
        
        for seq_id, seq_len in sorted_seqs:
            min_cp = min_cp_degrees.get(seq_id, 1)
            seq_capacity = seq_len  # Simplified: capacity = sequence length
            
            # Try to find best-fitting existing bin
            best_bin_idx = -1
            best_waste = float('inf')
            
            for bin_idx, group in enumerate(groups):
                # Check CP degree compatibility
                if min_cp > group['min_cp']:
                    continue
                # Check capacity constraint (simplified memory model)
                bin_capacity = group['min_cp'] * capacity_factor
                remaining = bin_capacity - group['used_capacity']
                if seq_capacity <= remaining:
                    waste = remaining - seq_capacity
                    if waste < best_waste:
                        best_waste = waste
                        best_bin_idx = bin_idx
            
            if best_bin_idx >= 0:
                # Pack into existing group
                groups[best_bin_idx]['seq_ids'].append(seq_id)
                groups[best_bin_idx]['total_len'] += seq_len
                groups[best_bin_idx]['seqlens'].append(seq_len)
                groups[best_bin_idx]['used_capacity'] += seq_capacity
                # Update min_cp if needed (take max for safety)
                groups[best_bin_idx]['min_cp'] = max(
                    groups[best_bin_idx]['min_cp'], min_cp
                )
            else:
                # Create new group/bin
                new_group = {
                    'seq_ids': [seq_id],
                    'min_cp': min_cp,
                    'total_len': seq_len,
                    'seqlens': [seq_len],
                    'used_capacity': seq_capacity
                }
                groups.append(new_group)
        
        return groups


    def dp_resource_allocation(self, groups, total_ranks, time_estimator, max_cp_degree=None,
                               cp_must_be_power_of_two=False):
        """
        Stage 2: Optimal Resource Assignment via 2D Dynamic Programming.

        Finds near-optimal CP degree assignment for each atomic group to minimize
        makespan (maximum execution time across all groups).

        Args:
            groups: list of groups from Stage 1
            total_ranks: total number of available ranks for CP
            time_estimator: function(seqlen, cp_size) -> estimated time
            max_cp_degree: optional upper bound on CP degree
            cp_must_be_power_of_two: restrict the enumerated CP values to
                powers of two (required for Ulysses CP so that num_attention_heads
                divides cp_size cleanly; disabled by default for Ring, which
                accepts arbitrary positive CPs).

        Returns:
            assignments: list of dicts with 'seq_ids', 'cp_degree', 'estimated_time'
        """
        if not groups:
            return []

        K = len(groups)
        if max_cp_degree is None:
            max_cp_degree = total_ranks

        def _candidate_cps(min_cp_g, upper):
            """Ordered list of legal CP values in [min_cp_g, upper]."""
            upper = min(upper, total_ranks)
            if cp_must_be_power_of_two:
                # Round min_cp_g UP to next power of two so no smaller-than-requested
                # cp is emitted; iterate powers of two up to `upper`.
                cp = 1
                while cp < min_cp_g:
                    cp <<= 1
                out = []
                while cp <= upper:
                    out.append(cp)
                    cp <<= 1
                return out
            return list(range(min_cp_g, upper + 1))

        # Precompute time estimates for each group at different CP degrees
        group_time_table = []
        for group in groups:
            times = {}
            total_seqlen = group['total_len']
            seqlens = group['seqlens']
            num_seqs = len(group['seq_ids'])
            avg_seqlen = total_seqlen / num_seqs if num_seqs > 0 else 0

            for cp in _candidate_cps(group['min_cp'], max_cp_degree):
                # Estimate group time: sum of individual sequence times
                # Simplified: use average sequence length scaled by group size
                effective_seqlen = int(total_seqlen)  # Total work
                t = time_estimator(seqlens, cp)
                times[cp] = t
            group_time_table.append(times)

        # DP initialization
        INF = float('inf')
        # DP[i][j]: min makespan for first i groups using exactly j ranks
        DP = [[INF] * (total_ranks + 1) for _ in range(K + 1)]
        Path = [[-1] * (total_ranks + 1) for _ in range(K + 1)]

        for i in range(total_ranks): DP[0][i] = 0

        # Fill DP table
        for i in range(1, K + 1):
            g_idx = i - 1
            min_cp = groups[g_idx]['min_cp']

            # Precompute min ranks needed for remaining groups (pruning)
            min_remaining = sum(groups[z]['min_cp'] for z in range(g_idx + 1, K))

            for j in range(total_ranks + 1):
                # Minimum ranks this group needs
                if j < min_cp:
                    continue

                # Maximum ranks we can assign to this group
                max_for_this = j - min_remaining
                # if max_for_this < min_cp:
                #     continue

                for cp in _candidate_cps(min_cp, min(max_cp_degree, j)):
                    prev_ranks = j - cp
                    if prev_ranks < 0 or DP[i-1][prev_ranks] == INF:
                        continue

                    current_time = group_time_table[g_idx].get(cp, INF)
                    # Makespan = max of previous groups and current group
                    makespan = max(DP[i-1][prev_ranks], current_time)

                    if makespan < DP[i][j]:
                        DP[i][j] = makespan
                        Path[i][j] = cp
        
        # Backtrack to recover optimal assignment
        assignments = []
        remaining_ranks = total_ranks
        
        for i in range(K, 0, -1):
            g_idx = i - 1
            cp = Path[i][remaining_ranks]
            
            # Fallback if no valid path found
            if cp == -1 or cp < groups[g_idx]['min_cp']:
                cp = groups[g_idx]['min_cp']
            
            assignments.append({
                'seq_ids': groups[g_idx]['seq_ids'],
                'cp_degree': cp,
                'estimated_time': group_time_table[g_idx].get(cp, 0)
            })
            remaining_ranks -= cp
            if remaining_ranks < 0:
                remaining_ranks = 0
        
        assignments.reverse()  # Restore original group order
        return assignments

    def compute_parallel_method(self, databatch):
        """
        Dynamic Hybrid Parallelism scheduler following DHP paper methodology.
        
        Implements two-stage approximation:
        1. Memory-aware Sequence Packing (Best-Fit Decreasing)
        2. Resource Allocation (2D Dynamic Programming)
        
        Args:
            databatch: dict containing 'input_ids', 'attention_mask', etc.
        
        Returns:
            group_list: list of CP degrees for each group
            data_list: list of sequence index lists assigned to each group
        """
        input_ids = databatch['input_ids']
        attention_mask = databatch['attention_mask']
        self.data_len = torch.sum(attention_mask, dim=1) + 1
        seq_len_chunk = self.seq_len_chunk

        # Compute sequence lengths [batch_size]
        data_len = torch.sum(attention_mask, dim=1) + 1
        batch_size = data_len.shape[0]

        if batch_size == 0:
            return [], []

        seq_ids = list(range(batch_size))
        seq_lens = data_len.tolist()

        # Calculate minimum CP degree for each sequence
        # Heuristic: longer sequences benefit more from parallelism.
        # CAP at max_cp_degree (== num_attention_heads for Ulysses) because
        # Ulysses cannot assign a CP group larger than the number of attention
        # heads.  Without this cap, a burst of 131073 tokens at chunk=8192 would
        # demand min_cp=17 which exceeds 2B's 16-head limit and forces DP to
        # fall back to 17, producing an invalid CP group.
        #
        # When cp_must_be_power_of_two (Ulysses), also round min_cp UP to the
        # next power of two — BFD uses min_cp to compute bin capacity and later
        # passes the resulting group.min_cp to DP, so both stages must agree on
        # a legal starting point.
        def _round_up_pow2(x):
            p = 1
            while p < x:
                p <<= 1
            return p

        max_available_cp = self.cluster_size // self.other_parallel_group_size
        min_cp_degrees = {}
        for seq_id, seqlen in zip(seq_ids, seq_lens):
            raw_min_cp = (seqlen + seq_len_chunk - 1) // seq_len_chunk
            mc = min(raw_min_cp, self.max_cp_degree)
            if self.cp_must_be_power_of_two and mc > 0:
                mc = min(_round_up_pow2(mc), self.max_cp_degree)
            min_cp_degrees[seq_id] = mc

        # Stage 1: Sequence Packing via Best-Fit Decreasing
        sequences = list(zip(seq_ids, seq_lens))
        groups = self.sequence_packing_bfd(
            sequences,
            min_cp_degrees,
            capacity_factor=seq_len_chunk  # Tunable parameter
        )

        # Fallback if packing fails
        if not groups:
            group_list = [1] * batch_size
            data_list = [[i] for i in range(batch_size)]
            return group_list, data_list

        # Stage 2: Resource Allocation via 2D Dynamic Programming
        total_available_ranks = max_available_cp
        assignments = self.dp_resource_allocation(
            groups,
            total_available_ranks,
            self.mock_time_estimator,
            max_cp_degree=min(self.max_cp_degree, total_available_ranks),
            cp_must_be_power_of_two=self.cp_must_be_power_of_two,
        )

        # Convert to output format
        group_list = [assign['cp_degree'] for assign in assignments]
        data_list = [assign['seq_ids'] for assign in assignments]

        # Safety check: ensure all sequences are assigned
        assigned_set = set()
        for dl in data_list:
            assigned_set.update(dl)

        unassigned = [i for i in range(batch_size) if i not in assigned_set]
        if unassigned:
            if data_list:
                # Assign to group with smallest current load
                min_load_idx = np.argmin([len(dl) for dl in data_list])
                data_list[min_load_idx].extend(unassigned)
            else:
                data_list.append(unassigned)
                group_list.append(1)

        # Rank-conservation invariant: update_rank_dicts allocates rank IDs
        # cumulatively from 0, so sum(group_list) must equal total_available_ranks
        # — otherwise high-indexed ranks end up unassigned and get_group_id() raises.
        # The DP solver sometimes leaves slack (e.g. burst saturates at
        # max_cp_degree while filler groups need fewer ranks than available).
        #
        # Two inflation strategies depending on which CP algo the caller uses:
        #   * Ring (cp_must_be_power_of_two=False, InternVL3 default): any
        #     positive integer CP is legal, so we grow groups by +1 in
        #     round-robin until the slack is absorbed.
        #   * Ulysses (cp_must_be_power_of_two=True, Qwen3VL): CP must divide
        #     num_attention_heads (== max_cp_degree for Qwen3VL's power-of-two
        #     head counts), so we only allow CP ← 2×CP doublings.
        delta = total_available_ranks - sum(group_list)
        if delta > 0:
            if self.cp_must_be_power_of_two:
                # Ulysses: round-robin doubling.
                made_progress = True
                while delta > 0 and made_progress:
                    made_progress = False
                    for i in range(len(group_list)):
                        if delta <= 0:
                            break
                        cur = group_list[i]
                        if cur <= 0 or cur >= self.max_cp_degree:
                            continue
                        next_cp = cur * 2
                        if next_cp > self.max_cp_degree:
                            continue
                        step = next_cp - cur  # = cur (one doubling step)
                        if step > delta:
                            continue
                        group_list[i] = next_cp
                        delta -= step
                        made_progress = True
            else:
                # Ring: round-robin +1 (any positive integer CP is legal).
                made_progress = True
                while delta > 0 and made_progress:
                    made_progress = False
                    for i in range(len(group_list)):
                        if delta <= 0:
                            break
                        if group_list[i] >= self.max_cp_degree:
                            continue
                        group_list[i] += 1
                        delta -= 1
                        made_progress = True

        if sum(group_list) != total_available_ranks:
            raise RuntimeError(
                f"[Scheduler] rank-conservation failed: sum(group_list)={sum(group_list)} "
                f"!= total_available_ranks={total_available_ranks}, "
                f"group_list={group_list}, max_cp_degree={self.max_cp_degree}, "
                f"cp_must_be_power_of_two={self.cp_must_be_power_of_two}. "
                f"Consider reducing MBS or increasing max_cp_degree."
            )

        return group_list, data_list

    def compute_parallel_method_naive(self, databatch):
        """
        Naive static-mesh scheduling: fixed CP size, data evenly distributed.

        All groups use the same CP degree (self.cp_window_size), mirroring the
        standard static CP mesh.  Samples are distributed round-robin across
        groups so that each group gets approximately batch_size / num_groups
        samples.  This serves as a baseline for verifying loss correctness
        against standard (non-hybrid) parallel training.

        Args:
            databatch: dict containing 'input_ids', 'attention_mask', etc.

        Returns:
            group_list: list of CP degrees for each group (all equal)
            data_list: list of sequence index lists assigned to each group
        """
        attention_mask = databatch['attention_mask']
        self.data_len = torch.sum(attention_mask, dim=1) + 1
        batch_size = self.data_len.shape[0]

        if batch_size == 0:
            return [], []

        fixed_cp = self.cp_window_size
        max_available_cp = self.cluster_size // self.other_parallel_group_size
        num_groups = max_available_cp // fixed_cp

        assert num_groups > 0, (
            f"Cannot form any group: max_available_cp={max_available_cp}, "
            f"fixed_cp={fixed_cp}"
        )
        assert max_available_cp % fixed_cp == 0, (
            f"max_available_cp ({max_available_cp}) must be divisible by "
            f"fixed_cp ({fixed_cp})"
        )

        # Round-robin distribute samples across groups
        data_list = [[] for _ in range(num_groups)]
        for i in range(batch_size):
            data_list[i % num_groups].append(i)

        group_list = [fixed_cp] * num_groups
        return group_list, data_list

    def get_data(self, databatch):
        data_layout = os.environ.get("DATA_LAYOUT", "TND").upper()
        new_databatch = {}
        rank_dict_local = self.rank_dicts[self.rank % self.other_parallel_group_size]
        local_rank_id = rank_dict_local[str(self.group_id)]
        local_data_id = self.data_dict[str(self.group_id)]
        # dict_keys(['pixel_values', 'image_flags', 'input_ids', 'labels', 'attention_mask', 'tranfer'])
        local_data_len = self.data_len[local_data_id]
        cp_size = len(local_rank_id)

        # --- Compute real and padded sequence lengths ---
        real_seqlens = local_data_len.tolist()                         # per-subsequence real token count
        padded_seqlens = [pad_len_to(s, cp_size * 2) for s in real_seqlens]  # pad each to cp_size*2 multiple

        # cu_seqlens must be standard cumulative-sum WITH leading 0: [0, len1, len1+len2, ...]
        padded_cumsum = [0]
        for p in padded_seqlens:
            padded_cumsum.append(padded_cumsum[-1] + p)

        total_padded_len = padded_cumsum[-1]
        max_padded_seqlen = max(padded_seqlens)

        # Two different cu_seqlens are needed for the MindSpeed ring attention path:
        #
        # 1. cu_seqlens_q (used by rotary embedding via _apply_rotary_pos_emb_thd):
        #    In MindSpeed, rotary is applied to the FULL packed tensor (before CP split).
        #    We store the actual padded_cumsum here. The patched rotary function detects
        #    that seqlens already sum to the tensor size and skips the // cp_size division.
        #
        # 2. cu_seqlens_q_padded (used by ring attention in AttentionWithCp.forward):
        #    Ring attention does `cu_seqlens_q_padded // cp_size` to get the per-rank
        #    chunk boundaries. This needs the real padded cumsum so that each chunk
        #    size = padded_seqlen_i / cp_size.
        cu_seqlens_q = torch.tensor(padded_cumsum, dtype=torch.int32).npu()
        cu_seqlens_q_padded = torch.tensor(padded_cumsum, dtype=torch.int32).npu()

        # --- Image patch mask ---
        # For models that don't produce image_flags in the batch (e.g. Qwen3VL via the
        # huggingface dataset pipeline), generate a synthetic all-ones tensor with one
        # entry per *merged* image patch, which equals the number of img_context_token_id
        # occurrences in input_ids.  This keeps img_token_per_patch == 1 and avoids the
        # integer-division underflow that occurs when image_flags length equals the number
        # of *raw* ViT patches (which can be spatial_merge_size^2 larger).
        if databatch.get('image_flags') is None:
            img_token_mask_tmp = torch.eq(databatch['input_ids'], self.img_context_token_id)
            n_img_tokens = int(img_token_mask_tmp.sum().item())
            ref_device = databatch['pixel_values'].device if databatch.get('pixel_values') is not None \
                else databatch['input_ids'].device
            databatch['image_flags'] = torch.ones(n_img_tokens, dtype=torch.long, device=ref_device)

        # Text-only fast path: no image tokens anywhere in the batch.  Skip the
        # image-patch mask computation (which would divide by zero on FloorDiv via
        # img_token_sum // sum(image_flags) when image_flags is empty) and leave
        # pv_patch_mask empty so the pixel_values/image_flags slicing below produces
        # None.  This is what enables synthetic text-only data to flow through.
        if databatch['image_flags'].numel() == 0:
            img_patch_mask = torch.zeros(0, dtype=torch.bool,
                                         device=databatch['image_flags'].device)
        else:
            img_token_mask = torch.eq(databatch['input_ids'], self.img_context_token_id)
            img_token_per_seq = torch.sum(img_token_mask, dim=-1)
            img_token_sum = torch.sum(img_token_per_seq)
            img_token_per_patch = img_token_sum // torch.sum(databatch['image_flags'])
            img_patch_cumsum = (img_token_per_seq // img_token_per_patch).cumsum(0)
            img_patch_cumsum_ = torch.zeros_like(img_patch_cumsum)
            img_patch_cumsum_[1:] = img_patch_cumsum[:-1]
            img_patch_mask = torch.zeros_like(databatch['image_flags'], dtype=torch.bool)
            for data_id in local_data_id:
                img_patch_mask[img_patch_cumsum_[data_id]:img_patch_cumsum[data_id]] = True

        # When pixel_values has more rows than image_flags (e.g. Qwen3VL where image_flags
        # counts merged patches but pixel_values contains raw ViT patches before spatial
        # merging), expand the mask so that each logical patch maps to expand_factor raw rows.
        pv = databatch.get('pixel_values')
        if pv is not None and img_patch_mask.shape[0] > 0 and pv.shape[0] > img_patch_mask.shape[0]:
            _pv_expand = pv.shape[0] // img_patch_mask.shape[0]
            pv_patch_mask = img_patch_mask.repeat_interleave(_pv_expand)
        else:
            pv_patch_mask = img_patch_mask

        # In two-phase (lightweight) mode pixel_values is None in the raw batch.
        # The pixel_loader callback will fill it after scheduling.
        _lightweight_mode = databatch.get('pixel_values') is None and self.pixel_loader is not None

        if data_layout == "TND":
            for key in databatch.keys():
                if key.startswith('_'):
                    # Private metadata keys (e.g. _image_path): skip, not tensors
                    continue
                if databatch[key] is None:
                    new_databatch[key] = databatch[key]
                    continue
                if key in ['attention_mask', 'labels', 'input_ids']:
                    value_tensor = databatch[key][local_data_id]
                    # Allocate padded TND buffer and place each subsequence at padded offsets
                    # For labels, padding must use -100 (IGNORE_INDEX) so that padded
                    # positions are excluded from loss computation. Using 0 would cause
                    # them to be treated as valid tokens (labels > -1), inflating
                    # token_nums and diluting the loss.
                    fill_value = -100 if key == 'labels' else 0
                    new_value_tensor = torch.full((total_padded_len,), fill_value, dtype=value_tensor.dtype, device=value_tensor.device)
                    seq_dim = value_tensor.shape[1] if value_tensor.dim() == 2 else value_tensor.shape[0]
                    for data_idx in range(value_tensor.shape[0]):
                        start = padded_cumsum[data_idx]
                        real_len = min(real_seqlens[data_idx], seq_dim)
                        new_value_tensor[start:start + real_len] = (
                            value_tensor[data_idx, :real_len]
                        )
                    new_databatch[key] = new_value_tensor.unsqueeze(0)
                elif key == 'pixel_values':
                    if _lightweight_mode:
                        new_databatch[key] = None  # Will be filled by pixel_loader below
                    else:
                        value_tensor = databatch[key][pv_patch_mask]
                        new_databatch[key] = value_tensor if value_tensor.shape[0] > 0 else None
                elif key == 'image_flags':
                    if _lightweight_mode:
                        new_databatch[key] = None
                    else:
                        value_tensor = databatch[key][img_patch_mask]
                        new_databatch[key] = value_tensor if value_tensor.shape[0] > 0 else None
                else:
                    value_tensor = databatch[key][local_data_id]
                    new_databatch[key] = value_tensor.contiguous()
            new_databatch["packed_seq_params"] = PackedSeqParams(
                qkv_format="thd",                           # must be 'thd', not 'tnd'
                cu_seqlens_q=cu_seqlens_q,                   # padded_cumsum * cp_size (for rotary)
                cu_seqlens_kv=cu_seqlens_q,
                cu_seqlens_q_padded=cu_seqlens_q_padded,     # padded_cumsum (for ring attention)
                cu_seqlens_kv_padded=cu_seqlens_q_padded,
                max_seqlen_q=max_padded_seqlen,
                max_seqlen_kv=max_padded_seqlen,
            )
        else:
            max_local_len = max_padded_seqlen
            for key in databatch.keys():
                if key.startswith('_'):
                    continue
                if databatch[key] is None:
                    new_databatch[key] = databatch[key]
                    continue
                if key in ['attention_mask', 'labels', 'input_ids']:
                    value_tensor = databatch[key][local_data_id]
                    bsz, pad_seq_len = value_tensor.shape if value_tensor.dim() == 2 else (1, value_tensor.shape[0])
                    if pad_seq_len < max_local_len:
                        fill_value = -100 if key == 'labels' else 0
                        new_value_tensor = torch.full([bsz, max_local_len], fill_value, dtype=value_tensor.dtype, device=value_tensor.device)
                        new_value_tensor[:, :pad_seq_len] = value_tensor
                        value_tensor = new_value_tensor
                    else:
                        value_tensor = value_tensor[:, :max_local_len]
                    new_databatch[key] = value_tensor
                elif key in ['pixel_values', 'image_flags']:
                    if _lightweight_mode:
                        new_databatch[key] = None  # Will be filled by pixel_loader below
                    else:
                        value_tensor = databatch[key][img_patch_mask]
                        new_databatch[key] = value_tensor if value_tensor.shape[0] > 0 else None
                else:
                    value_tensor = databatch[key][local_data_id]
                    new_databatch[key] = value_tensor.contiguous()

        # Two-phase loading: load pixel_values for only this rank's assigned samples
        if _lightweight_mode:
            pv, flags = self.pixel_loader(databatch, local_data_id)
            # Move to the same device/dtype as the rest of the batch.
            # The batch was already moved to NPU+bf16 by move_to_device() in pretrain_vlm.py.
            ref = databatch.get('input_ids')
            if ref is None:
                ref = databatch.get('attention_mask')
            if ref is not None and pv is not None:
                pv = pv.to(device=ref.device, dtype=torch.bfloat16)
                flags = flags.to(device=ref.device)
            new_databatch['pixel_values'] = pv
            new_databatch['image_flags'] = flags

        return new_databatch

    @timing
    def next_batch(self, databatch):
        self.data_dict = {}
        if self.schedule_mode == "naive":
            group_list, data_list = self.compute_parallel_method_naive(databatch)
        else:
            group_list, data_list = self.compute_parallel_method(databatch)

        self.update_rank_dicts(group_list)
        self.update_group_data_id(data_list)
        self.update_parallel_group()

        return self.get_data(databatch)


# ---------------------------------------------------------------------------
#  Double-Buffered Async Scheduler
# ---------------------------------------------------------------------------

@dataclass
class _ScheduleBuffer:
    """Container for pre-computed scheduling results."""
    group_list: List[int] = field(default_factory=list)
    data_list: List[List[int]] = field(default_factory=list)
    rank_dicts: List[Dict] = field(default_factory=list)
    data_dict: Dict[str, Any] = field(default_factory=dict)
    data_len: Optional[torch.Tensor] = None
    group_id: Optional[str] = None
    cp_group: Optional[Any] = None
    local_group_ranks: Optional[List[int]] = None
    prepared_data: Optional[Dict] = None
    ready: bool = False

    def reset(self):
        self.group_list = []
        self.data_list = []
        self.rank_dicts = []
        self.data_dict = {}
        self.data_len = None
        self.group_id = None
        self.cp_group = None
        self.local_group_ranks = None
        self.prepared_data = None
        self.ready = False


class DoubleBufferedScheduler(Scheduler):
    """Scheduler with double-buffered async prefetch.

    While the current step runs forward/backward on the GPU, a background
    thread pre-computes the scheduling decision, builds rank dicts, resolves
    the process group, and prepares data tensors for the *next* step on a
    separate CUDA stream.  At the sync point the main thread only needs to
    do O(1) pointer assignments.

    Usage in the training loop::

        # First step — synchronous cold start
        batch_0 = scheduler.next_batch(raw_batch_0)

        # Kick off prefetch for step 1
        scheduler.start_prefetch(raw_batch_1)

        forward_backward(batch_0)

        # Consume prefetch result (near-instant) and kick off step 2
        batch_1 = scheduler.swap_and_get_data()
        scheduler.start_prefetch(raw_batch_2)

        forward_backward(batch_1)
        ...
    """

    def __init__(self, *args, warmup_steps: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffers = [_ScheduleBuffer(), _ScheduleBuffer()]
        self._active = 0  # index of the buffer currently used by forward
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._warmup_steps = warmup_steps
        self._step_count = 0
        self._prefetch_error: Optional[Exception] = None

    @property
    def _next_buf(self) -> _ScheduleBuffer:
        return self._buffers[1 - self._active]

    def _ensure_stream(self):
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream()

    # ── Pure-function helpers (no self mutation) ──

    @staticmethod
    def _build_rank_dicts_pure(group_list, other_parallel_group_size, rank):
        """Build rank_dicts without barrier or self mutation."""
        rank_dicts = [{} for _ in range(other_parallel_group_size)]
        cumsum = 0
        for group_id, group_size in enumerate(group_list):
            group_ranks = [(i + cumsum) * other_parallel_group_size
                           for i in range(group_size)]
            for op_id in range(other_parallel_group_size):
                rank_dicts[op_id][str(group_id)] = [
                    gr + op_id for gr in group_ranks
                ]
            cumsum += group_size
        return rank_dicts

    def _resolve_group_id_pure(self, rank_dicts):
        """Find this rank's group_id from rank_dicts."""
        rank_dict_local = rank_dicts[self.rank % self.other_parallel_group_size]
        for gid in rank_dict_local:
            if self.rank in rank_dict_local[gid]:
                return gid
        raise ValueError(f"Rank {self.rank} not in any group")

    def _prepare_data_pure(self, databatch, data_dict, data_len, group_id,
                           local_group_ranks, rank_dicts):
        """Pure-function version of get_data(): no self reads except immutable config."""
        data_layout = os.environ.get("DATA_LAYOUT", "TND").upper()
        new_databatch = {}
        local_data_id = data_dict[str(group_id)]
        local_data_len = data_len[local_data_id]
        cp_size = len(local_group_ranks)

        # --- Compute real and padded sequence lengths ---
        real_seqlens = local_data_len.tolist()
        padded_seqlens = [pad_len_to(s, cp_size * 2) for s in real_seqlens]

        padded_cumsum = [0]
        for p in padded_seqlens:
            padded_cumsum.append(padded_cumsum[-1] + p)

        total_padded_len = padded_cumsum[-1]
        max_padded_seqlen = max(padded_seqlens)

        cu_seqlens_q = torch.tensor(padded_cumsum, dtype=torch.int32,
                                    device=databatch['input_ids'].device)
        cu_seqlens_q_padded = torch.tensor(padded_cumsum, dtype=torch.int32,
                                           device=databatch['input_ids'].device)

        # --- Image patch mask ---
        # Synthesise image_flags when absent (e.g. Qwen3VL / huggingface dataset pipeline).
        # One flag per *merged* image patch == one per img_context_token_id occurrence.
        if databatch.get('image_flags') is None:
            img_token_mask_tmp = torch.eq(databatch['input_ids'], self.img_context_token_id)
            n_img_tokens = int(img_token_mask_tmp.sum().item())
            ref_device = databatch['pixel_values'].device if databatch.get('pixel_values') is not None \
                else databatch['input_ids'].device
            databatch['image_flags'] = torch.ones(n_img_tokens, dtype=torch.long, device=ref_device)

        # Text-only fast path: no image tokens anywhere in the batch.  Skip the
        # image-patch mask computation (which would divide by zero on FloorDiv via
        # img_token_sum // sum(image_flags) when image_flags is empty) and leave
        # pv_patch_mask empty so the pixel_values/image_flags slicing below produces
        # None.  This is what enables synthetic text-only data to flow through.
        if databatch['image_flags'].numel() == 0:
            img_patch_mask = torch.zeros(0, dtype=torch.bool,
                                         device=databatch['image_flags'].device)
        else:
            img_token_mask = torch.eq(databatch['input_ids'], self.img_context_token_id)
            img_token_per_seq = torch.sum(img_token_mask, dim=-1)
            img_token_sum = torch.sum(img_token_per_seq)
            img_token_per_patch = img_token_sum // torch.sum(databatch['image_flags'])
            img_patch_cumsum = (img_token_per_seq // img_token_per_patch).cumsum(0)
            img_patch_cumsum_ = torch.zeros_like(img_patch_cumsum)
            img_patch_cumsum_[1:] = img_patch_cumsum[:-1]
            img_patch_mask = torch.zeros_like(databatch['image_flags'], dtype=torch.bool)
            for data_id in local_data_id:
                img_patch_mask[img_patch_cumsum_[data_id]:img_patch_cumsum[data_id]] = True

        # Expand mask when pixel_values has more rows than image_flags (e.g. Qwen3VL).
        pv = databatch.get('pixel_values')
        if pv is not None and img_patch_mask.shape[0] > 0 and pv.shape[0] > img_patch_mask.shape[0]:
            _pv_expand = pv.shape[0] // img_patch_mask.shape[0]
            pv_patch_mask = img_patch_mask.repeat_interleave(_pv_expand)
        else:
            pv_patch_mask = img_patch_mask

        if data_layout == "TND":
            for key in databatch.keys():
                if databatch[key] is None:
                    new_databatch[key] = databatch[key]
                    continue
                if key in ['attention_mask', 'labels', 'input_ids']:
                    value_tensor = databatch[key][local_data_id]
                    fill_value = -100 if key == 'labels' else 0
                    new_value_tensor = torch.full(
                        (total_padded_len,), fill_value,
                        dtype=value_tensor.dtype, device=value_tensor.device)
                    seq_dim = value_tensor.shape[1] if value_tensor.dim() == 2 else value_tensor.shape[0]
                    for data_idx in range(value_tensor.shape[0]):
                        start = padded_cumsum[data_idx]
                        real_len = min(real_seqlens[data_idx], seq_dim)
                        new_value_tensor[start:start + real_len] = (
                            value_tensor[data_idx, :real_len]
                        )
                    new_databatch[key] = new_value_tensor.unsqueeze(0)
                elif key == 'pixel_values':
                    value_tensor = databatch[key][pv_patch_mask]
                    new_databatch[key] = value_tensor if value_tensor.shape[0] > 0 else None
                elif key == 'image_flags':
                    value_tensor = databatch[key][img_patch_mask]
                    new_databatch[key] = value_tensor if value_tensor.shape[0] > 0 else None
                else:
                    new_databatch[key] = databatch[key][local_data_id].contiguous()
            new_databatch["packed_seq_params"] = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_q,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_q_padded,
                max_seqlen_q=max_padded_seqlen,
                max_seqlen_kv=max_padded_seqlen,
            )
        else:
            max_local_len = max_padded_seqlen
            for key in databatch.keys():
                if databatch[key] is None:
                    new_databatch[key] = databatch[key]
                    continue
                if key in ['attention_mask', 'labels', 'input_ids']:
                    value_tensor = databatch[key][local_data_id]
                    bsz, pad_seq_len = (value_tensor.shape if value_tensor.dim() == 2
                                        else (1, value_tensor.shape[0]))
                    if pad_seq_len < max_local_len:
                        fill_value = -100 if key == 'labels' else 0
                        new_value_tensor = torch.full(
                            [bsz, max_local_len], fill_value,
                            dtype=value_tensor.dtype, device=value_tensor.device)
                        new_value_tensor[:, :pad_seq_len] = value_tensor
                        value_tensor = new_value_tensor
                    else:
                        value_tensor = value_tensor[:, :max_local_len]
                    new_databatch[key] = value_tensor
                elif key in ['pixel_values', 'image_flags']:
                    value_tensor = databatch[key][img_patch_mask]
                    new_databatch[key] = value_tensor if value_tensor.shape[0] > 0 else None
                else:
                    new_databatch[key] = databatch[key][local_data_id].contiguous()

        return new_databatch

    # ── Background worker ──

    def _prefetch_worker(self, databatch):
        """Background thread: schedule + build groups + prepare data."""
        buf = self._next_buf
        buf.reset()
        try:
            # Stage 1: Scheduling algorithm (pure CPU)
            if self.schedule_mode == "naive":
                buf.group_list, buf.data_list = self.compute_parallel_method_naive(databatch)
            else:
                buf.group_list, buf.data_list = self.compute_parallel_method(databatch)
            # data_len was set as side effect; capture it into buffer
            buf.data_len = self.data_len.clone()

            # Stage 2: Build rank_dicts (pure CPU, no barrier)
            buf.rank_dicts = self._build_rank_dicts_pure(
                buf.group_list, self.other_parallel_group_size, self.rank)
            buf.data_dict = {str(i): d for i, d in enumerate(buf.data_list)}

            # Stage 3: Resolve group assignment (pure CPU)
            buf.group_id = self._resolve_group_id_pure(buf.rank_dicts)
            rank_dict_local = buf.rank_dicts[self.rank % self.other_parallel_group_size]
            buf.local_group_ranks = rank_dict_local[str(buf.group_id)]

            # Stage 4: Lookup process group from cache (no create_group allowed here).
            # All groups must have been created during warmup (synchronous path).
            # Cache miss → raises RuntimeError, caught below as _prefetch_error.
            buf.cp_group = self.get_from_group_pool(buf.local_group_ranks, allow_create=False)

            # Stage 5: Prepare data tensors (on separate CUDA stream)
            self._ensure_stream()
            with torch.cuda.stream(self._prefetch_stream):
                buf.prepared_data = self._prepare_data_pure(
                    databatch, buf.data_dict, buf.data_len, buf.group_id,
                    buf.local_group_ranks, buf.rank_dicts)

            buf.ready = True

        except Exception as e:
            self._prefetch_error = e

    # ── Public API ──

    def start_prefetch(self, databatch):
        """Launch background prefetch for the next step.

        Call this right after obtaining the raw batch from the dataloader
        and moving it to device.  The prefetch runs concurrently with the
        current step's forward/backward.
        """
        # Wait for any previous prefetch to finish (shouldn't normally happen)
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None

        self._prefetch_error = None
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, args=(databatch,),
            daemon=True)
        self._prefetch_thread.start()

    def swap_and_get_data(self):
        """Consume the prefetched result: sync stream, swap mpu globals, return data.

        This is the only sync point on the critical path.  Cost ≈
        stream.synchronize() + a handful of pointer assignments.
        """
        # Wait for background thread
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None

        # Check for errors in the worker
        if self._prefetch_error is not None:
            raise RuntimeError(
                f"Prefetch worker failed: {self._prefetch_error}"
            ) from self._prefetch_error

        buf = self._next_buf
        assert buf.ready, "Prefetch buffer not ready"

        # Synchronize the CUDA stream used by _prepare_data_pure
        if self._prefetch_stream is not None:
            self._prefetch_stream.synchronize()

        # ═══ O(1) pointer assignments — the only blocking work ═══
        mpu._CONTEXT_PARALLEL_GROUP = buf.cp_group
        mpu._CONTEXT_PARALLEL_GLOBAL_RANKS = list(buf.local_group_ranks)
        ms_mpu._CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV = list(buf.local_group_ranks)
        ms_mpu._CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV = list(buf.local_group_ranks)

        # Sync Scheduler's own state (so get_num_groups() etc. still work)
        self.rank_dicts = buf.rank_dicts
        self.data_dict = buf.data_dict
        self.data_len = buf.data_len
        self.group_id = buf.group_id
        self.hybrid_group = buf.cp_group

        # Flip active buffer
        self._active = 1 - self._active
        self._step_count += 1

        return buf.prepared_data

    def is_warmup(self):
        """True during the first few steps when we run synchronously."""
        return self._step_count < self._warmup_steps

    def next_batch(self, databatch):
        """Synchronous fallback — used during warmup and as the original API."""
        result = super().next_batch(databatch)
        self._step_count += 1
        return result
