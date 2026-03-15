import torch
import psutil
import os
import gc
import time
import numpy as np

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
                 schedule_mode="dynamic"):
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

    def get_from_group_pool(self, rank_list):
        group_uid = self.get_group_uid(rank_list)
        if group_uid in self.group_pool.keys():
            return self.group_pool[group_uid]
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


    def dp_resource_allocation(self, groups, total_ranks, time_estimator, max_cp_degree=None):
        """
        Stage 2: Optimal Resource Assignment via 2D Dynamic Programming.
        
        Finds near-optimal CP degree assignment for each atomic group to minimize
        makespan (maximum execution time across all groups).
        
        Args:
            groups: list of groups from Stage 1
            total_ranks: total number of available ranks for CP
            time_estimator: function(seqlen, cp_size) -> estimated time
            max_cp_degree: optional upper bound on CP degree
        
        Returns:
            assignments: list of dicts with 'seq_ids', 'cp_degree', 'estimated_time'
        """
        if not groups:
            return []
        
        K = len(groups)
        if max_cp_degree is None:
            max_cp_degree = total_ranks
        
        # Precompute time estimates for each group at different CP degrees
        group_time_table = []
        for group in groups:
            times = {}
            total_seqlen = group['total_len']
            seqlens = group['seqlens']
            num_seqs = len(group['seq_ids'])
            avg_seqlen = total_seqlen / num_seqs if num_seqs > 0 else 0
            
            for cp in range(group['min_cp'], min(max_cp_degree + 1, total_ranks + 1)):
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
                
                for cp in range(min_cp, min(max_cp_degree, j) + 1):
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

    # def compute_parallel_method(self, databatch):
    #     # Toy scheduler for simple test
    #     input_ids = databatch['input_ids']
    #     attention_mask = databatch['attention_mask']
    #     self.data_len = torch.sum(attention_mask, dim=1) + 1
    #     sorted_data_id = torch.argsort(self.data_len)
    #     # group_list = [1, 1, 1, 1, 2, 2]
    #     # data_list = [[sorted_data_id[0].tolist()], [sorted_data_id[1].tolist()], [sorted_data_id[2].tolist()], [sorted_data_id[3].tolist()], sorted_data_id[4:6].tolist(), sorted_data_id[6:8].tolist()]
    #     group_list = [1, 1, 1, 2, 3]
    #     data_list = [[sorted_data_id[0].tolist()], [sorted_data_id[1].tolist()], [sorted_data_id[2].tolist()], sorted_data_id[3:5].tolist(), sorted_data_id[5:8].tolist()]
    #     # group_list = [1, 1, 1, 1, 4]
    #     # data_list = [[sorted_data_id[0].tolist()], [sorted_data_id[1].tolist()], [sorted_data_id[2].tolist()], [sorted_data_id[3].tolist()], sorted_data_id[4:8].tolist()]
    #     return group_list, data_list

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
        seq_len_chunk = 4096
        
        # Compute sequence lengths [batch_size]
        data_len = torch.sum(attention_mask, dim=1) + 1
        batch_size = data_len.shape[0]
        
        if batch_size == 0:
            return [], []
        
        seq_ids = list(range(batch_size))
        seq_lens = data_len.tolist()
        
        # Calculate minimum CP degree for each sequence
        # Heuristic: longer sequences benefit more from parallelism
        max_available_cp = self.cluster_size // self.other_parallel_group_size
        min_cp_degrees = {}
        for seq_id, seqlen in zip(seq_ids, seq_lens):
            min_cp_degrees[seq_id] = (seqlen + seq_len_chunk - 1) // seq_len_chunk

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
            max_cp_degree=min(8, total_available_ranks)
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
        data_layout = "TND"
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

        # --- Image patch mask (unchanged logic) ---
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

        if data_layout == "TND":
            for key in databatch.keys():
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
                elif key in ['pixel_values', 'image_flags']:
                    value_tensor = databatch[key][img_patch_mask]
                    new_databatch[key] = value_tensor
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
                    value_tensor = databatch[key][img_patch_mask]
                    new_databatch[key] = value_tensor
                else:
                    value_tensor = databatch[key][local_data_id]
                    new_databatch[key] = value_tensor.contiguous()

        return new_databatch
    
    @timing
    def next_batch(self, databatch):
        if torch.distributed.get_rank() == 0:
            print("Execute Next Batch Function")
        self.data_dict = {}
        if self.schedule_mode == "naive":
            group_list, data_list = self.compute_parallel_method_naive(databatch)
        else:
            group_list, data_list = self.compute_parallel_method(databatch)

        self.update_rank_dicts(group_list)
        self.update_group_data_id(data_list)
        self.update_parallel_group()

        return self.get_data(databatch)
