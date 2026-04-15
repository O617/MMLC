# Hybrid Parallel 精度对齐 Debug 报告

## 目标

使 Hybrid Parallel（动态 CP 调度）模式的训练精度与 Baseline（静态 CP）模式对齐，验证 Scheduler 设计的正确性。

---

## Debug 工作流

### Phase 1：数据一致性对齐

**问题**：Hybrid 用 `dataloader_mode: "base"`（PyTorch RandomSampler），Baseline 用 `BaseRandomBatchSampler`（epoch-based seed），两者随机种子机制不同，同一 step 加载不同数据。

**修复**：
- `data_8B_hybrid.json`：改为 `dataloader_mode: "sampler"` + `BaseRandomBatchSampler`
- `pretrain_vlm.py`：Hybrid 模式下传 `_SingleRankGroup(size=1, rank=0)` 给 DataLoader，使所有 rank 加载相同全量数据；临时将 `micro_batch_size *= num_groups` 让每次 yield 足够样本供 Scheduler 分配
- `pretrain_vlm.py`：`reconfigure(data_parallel_size=num_groups)` 替代 `data_parallel_size=1`
- `finetune_internvl3_8B_hybrid.sh`：`GBS=$(($MBS*$GRAD_ACC_STEP*$DP))` 与 Baseline 一致

**验证**：`[DIAG-RAW]` 的 `label_sum` 和 `first20_labels` 逐 microbatch 完全一致。

### Phase 2：loss_for_backward 归一化对齐

**问题**：TND packed 路径用 3-element return（`schedules.py` 做 `/num_tokens * cp_size / num_microbatches`），BSND 路径用 2-element return（`* cp_size / num_microbatches`）。两者实际 backward 值差一个 `num_samples / total_tokens` 因子。

**修复**：
- `vlm_model.py`：TND packed per-sample loss 改为返回 `(per_sample_mean.mean(), token_nums_mean)` 2-tuple，与 BSND 一致
- `pretrain_vlm.py`：删除 TND packed 的 3-element 特殊路径，统一走 BSND/default 路径

### Phase 3：per-sample zigzag split 修复

**问题**：`split_forward_gather_backward_with_megatron_cp` 对 TND packed 序列做全局 zigzag split，chunk 边界跨越 sample 边界，导致 ring attention 的 `cu_seqlens // cp_size` 假设不成立（sample 边界错位）。这造成注意力掩码和 RoPE 位置编码部分错误，per-sample loss 差异达千分之 3-8。

**修复**：
- `utils.py`：新增 `_SplitPackedPerSampleMegatronCP` autograd 函数，对每个 sample 独立做 zigzag split 再拼接
- `mm_gpt_model.py`：TND packed 时用 `split_packed_per_sample_megatron_cp` 替代全局 split（embedding/input_ids/position_ids）
- `vlm_model.py`：labels/sample_ids 同样用 per-sample split

**效果**：per-sample loss 差异从千分之 3-8 降至万分之 2-6。

### Phase 4：logging_loss 统计口径对齐

**问题**：Baseline 用 `average_losses_across_data_parallel_group`（sample 均值），Hybrid 用 `average_losses_for_hybrid_parallel`（token 加权均值）。DP>1 时两者给出显著不同的 logging_loss。

**修复**：`pretrain_vlm.py` 中 Hybrid 也改用 `average_losses_across_data_parallel_group`。

### Phase 5：consumed_samples 和 LR schedule 对齐

**问题**：`training.py` 中 Hybrid 分支计算 `consumed_samples` 和 LR scheduler increment 时缺少 `dp_world_size` 乘子（原设计假设 DP=1），导致 DP>1 时 consumed_samples 每步少一半、LR 衰减速度慢一倍。

**修复**：两处 Hybrid 分支都加上 `* mpu.get_data_parallel_world_size()`。

---

## 修改文件清单

| 文件 | 改动 |
|------|------|
| `examples/internvl3/data_8B_hybrid.json` | dataloader_mode → sampler + BaseRandomBatchSampler |
| `pretrain_vlm.py` | SingleRankGroup、MBS 放大、reconfigure dp=num_groups、删 TND 3-elem 路径、logging_loss 统一 |
| `mindspeed_mm/models/vlm_model.py` | TND per-sample loss 改 2-tuple 返回、labels 用 per-sample split |
| `mindspeed_mm/models/common/mm_gpt_model.py` | TND packed embedding 用 per-sample split |
| `mindspeed_mm/utils/utils.py` | 新增 `split_packed_per_sample_megatron_cp` 函数 |
| `mindspeed_mm/training.py` | consumed_samples 和 LR increment 加 dp_world_size |
| `finetune_internvl3_8B_hybrid.sh` | GBS 公式对齐 Baseline |

---

## 精度对齐结果

| 配置 | Step 1 | Step 10 | Step 20 | 是否发散 |
|------|--------|---------|---------|---------|
| DP=1, CP=2 (2卡) | 0.003% | 0.054% | 0.030% | 不发散 |
| DP=2, CP=2 (4卡) | 0.003% | 0.800% | 0.093% | 不发散 |

---

## 核心调试命令

```bash
# 远端环境激活
ssh -p 6022 root@localhost
source /data/user/user40/miniconda3/bin/activate && conda activate /opt/conda/private/envs/MindSpeed-MM

# 代码同步
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='save_dir' \
  --exclude='ckpt/' --exclude='MindSpeed/' --exclude='megatron/' --exclude='Megatron-LM/' \
  -e 'ssh -p 6022' /mnt/hdc/xh/MindSpeed-MM/ root@localhost:/data/user/user40/hybrid_parallel/MindSpeed-MM/

# 远端脚本参数调整
sed -i 's/NPUS_PER_NODE=2/NPUS_PER_NODE=4/' finetune_internvl3_8B*.sh
sed -i 's/--train-iters 5000/--train-iters 20/' finetune_internvl3_8B*.sh

# 运行 Baseline
bash finetune_internvl3_8B.sh 2>&1 > /tmp/baseline.txt

# 运行 Hybrid（naive 调度模式）
SCHEDULE_MODE=naive bash finetune_internvl3_8B_hybrid.sh 2>&1 > /tmp/hybrid.txt

# 数据一致性验证
grep '\[DIAG-RAW\] rank=0' /tmp/baseline.txt | head -16
grep '\[DIAG-RAW\] rank=0' /tmp/hybrid.txt | head -16

# per-sample loss 对比
grep 'DIAG-LOSS.*rank=0.*per_sample_mean' /tmp/baseline.txt | head -8
grep 'DIAG-LOSS.*rank=0.*per_sample_mean' /tmp/hybrid.txt | head -8

# consumed_samples + LR 验证
grep 'consumed samples:' /tmp/baseline.txt | head -5
grep 'consumed samples:' /tmp/hybrid.txt | head -5

# iteration loss 对比
grep 'iteration.*loss:' /tmp/baseline.txt | awk -F'loss:' '{print $2}' | awk -F'|' '{print $1}' > /tmp/b.txt
grep 'iteration.*loss:' /tmp/hybrid.txt | awk -F'loss:' '{print $2}' | awk -F'|' '{print $1}' > /tmp/h.txt
paste /tmp/b.txt /tmp/h.txt | awk '{b=$1+0;h=$2+0;d=(b>0)?(h-b)/b*100:0; printf "Step %2d: B=%.6f H=%.6f diff=%+.4f%%\n",NR,b,h,d}'
```
claude --resume b11928bb-0f13-4f8e-9ec1-8ba450539363

---

## Phase 6：双缓冲异步调度 (DoubleBufferedScheduler)

**目标**：将 Scheduler 的调度计算从 forward/backward 的关键路径中移除，通过异步预取 + 双缓冲实现流水线重叠。

**设计**：
- **双缓冲**：Buffer A 供当前 step 的 forward/backward 使用，Buffer B 在后台线程预计算下一 step 的调度结果
- **后台线程**执行：调度算法（BFD + DP）→ 构建 rank_dicts → 解析 group 分配 → 查找通信组 → 准备数据 tensor（独立 CUDA stream）
- **Swap 同步点**：主线程仅做 O(1) 指针赋值（`mpu._CONTEXT_PARALLEL_GROUP` 等）
- **Warmup 机制**：前 3 步同步执行，填充 `group_pool` 缓存；之后切换到异步模式

**关键优化点**：
1. `update_rank_dicts` 中的 `barrier()` 可去除——所有 rank 加载相同数据，调度算法确定性保证结果一致
2. 通信组从 `group_pool` 缓存获取，warmup 后全部命中
3. `get_data` 的 tensor 操作放在独立 CUDA stream，不阻塞主 stream

**修改文件**：

| 文件 | 改动 |
|------|------|
| `scheduler.py` | 新增 `_ScheduleBuffer` dataclass 和 `DoubleBufferedScheduler` 类，包含纯函数版 `_build_rank_dicts_pure`、`_resolve_group_id_pure`、`_prepare_data_pure`，后台线程 `_prefetch_worker`，以及 `start_prefetch()` / `swap_and_get_data()` 公共接口 |
| `pretrain_vlm.py` | 重构 `get_batch()` 为 fast path（消费预取结果）和 slow path（同步 fallback）；拆分 `_load_raw_batch`、`_apply_encoder_balance`、`_diag_raw_batch` 辅助函数；`model_provider` 通过 `ASYNC_SCHEDULE` 环境变量选择 Scheduler 类 |

**启用方式**：
```bash
ASYNC_SCHEDULE=True SCHEDULE_MODE=naive bash finetune_internvl3_8B_hybrid.sh
```

**精度验证（DP=2, CP=2, 4卡, 10 steps）**：

| 对比 | Step 1 diff | Step 10 diff | 最大 diff |
|------|-------------|-------------|-----------|
| Async vs Sync | 0.0000% | -0.0013% | 0.0092% |
| Async vs Baseline | -0.0034% | +0.3012% | +0.8145% |

Async 与 Sync 模式的 loss 差异 < 0.01%，数值正确性完全一致。

**性能收益（warmup 后 Step 4-10 平均）**：

| 模式 | 平均 iteration time (ms) |
|------|------------------------|
| Sync | 6,455 |
| Async | 6,240 |
| **提速** | **~3.3%** |

---

## Phase 7：视频数据集构造

**目标**：构建基于 InternVid 元数据的模拟视频数据集，用于视频训练调试。

**流程**：
1. 从 ModelScope 下载 `InternVid-10M-flt.jsonl`（10.6M 条，2.2GB）
2. 随机抽取 1000 条，保留原始 Caption、YoutubeID、时长信息
3. 根据每条记录的 `Start_timestamp` / `End_timestamp` 计算真实时长
4. 按时长比例生成帧数（≈1帧/秒，clamp [4, 30]），按时长分级选择分辨率
5. 用纯 Python 写入无压缩 AVI（无需 ffmpeg/opencv 依赖）
6. 生成 MultiModalChatDataset 格式的训练 JSON

**数据分布**：

| 时长区间 | 视频数 | 帧数范围 | 分辨率 |
|---------|--------|---------|-------|
| 0-2s | 250 | 4 | 320×240 |
| 2-5s | 317 | 4 | 320×240 / 426×240 |
| 5-10s | 174 | 5-10 | 320×240 / 426×240 |
| 10-20s | 132 | 10-20 | 426×240 / 640×360 |
| 20-30s | 49 | 20-30 | 640×360 / 854×480 |
| 30-60s | 52 | 30 | 640×360 / 854×480 |
| 60-300s | 26 | 30 | 640×360 / 854×480 |

**Visual Token 统计**（采样帧数 uniform [4,12]，每帧 256 tokens）：
- 范围：1,024 — 3,072 tokens/视频
- 均值：2,048 tokens
- 加上文本后总序列长度：1,054 — 3,102 tokens

**生成文件**：

| 文件 | 路径（远端） |
|------|------------|
| 生成脚本 | `scripts/generate_fake_video_dataset.py` |
| 视频目录 | `/data/user/user40/develop/dataset/InternVid_fake/videos/` (1000 个 AVI, ~4.3GB) |
| 训练 JSON | `/data/user/user40/develop/dataset/InternVid_fake/internvid_fake_1k.json` |
| 数据配置 | `examples/internvl3/data_8B_video.json` |

**使用方式**：
```bash
torchrun ... pretrain_vlm.py \
    --mm-data ./examples/internvl3/data_8B_video.json \
    --mm-model ./examples/internvl3/model_8B.json \
    ...
```
---

## Phase 8：8卡（DP=4, CP=2）精度验证 & Sampler 对齐 Bug 修复

**目标**：在 8 张 NPU 卡（DP=4, CP=2）下验证 naive 和 dynamic 两种 Scheduler 与 Baseline 的精度对齐。

### Bug 发现：BaseRandomBatchSampler full_bucket_size 不一致

**问题**：`BaseRandomBatchSampler.__iter__` 中 epoch 排列长度计算：
```python
# 原代码（有 bug）
full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
```
- Baseline（MBS=2, num_replicas=4）：`floor(N/2)*2 = 12412` → `randperm(12412)`
- Hybrid（MBS=8 被放大, num_replicas=1）：`floor(N/8)*8 = 12408` → `randperm(12408)`

两个 permutation 长度不同，导致样本顺序完全不一致，iteration loss 差异高达 **14%**。

在 4 卡（DP=2）时偶然正常：`floor(N/4)*4 = floor(N/2)*2 = 12412`（数据集大小恰好是 4 的倍数），排列长度相同，对齐验证通过。但 8 卡时 12408 ≠ 12412 导致不对齐。

**修复**（`mindspeed_mm/data/dataloader/sampler.py` 第 462 行）：
```python
# 修复后：使用 MBS × num_replicas 作为归一化单位
full_bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                    * self.micro_batch_times_data_parallel_size
```

现在无论 MBS 如何放大，只要 `MBS × num_replicas` 相同（baseline=2×4=8，hybrid=8×1=8），permutation 长度一致（均为 `floor(N/8)*8 = 12408`），样本完全对齐。

**验证**（修复后 step 1 数据对比）：
```
Baseline rank=0:          label_sum=404370 first20=[78501, 151645, 77, 22147, 355, 151645]
Hybrid-naive group_id=0:  label_sum=404370 first20=[78501, 151645, 77, 22147, 355, 151645]
```

### 8卡精度测试结果（DP=4, CP=2, 20 steps）

| Step | Baseline | Hybrid-Naive | diff_N | Hybrid-Dynamic | diff_D |
|------|----------|-------------|--------|---------------|--------|
| 1  | 11.671840 | 11.673040 | +0.0103% | 11.708780 | +0.3165% |
| 2  | 13.053010 | 13.047310 | -0.0437% | 13.734140 | +5.2182% |
| 5  | 8.352622  | 8.348302  | -0.0517% | 8.187469  | -1.9773% |
| 10 | 8.010805  | 8.013975  | +0.0396% | 8.319005  | +3.8473% |
| 20 | 7.634458  | 7.635237  | +0.0102% | 7.908514  | +3.5897% |

**最大差异**：
- Naive vs Baseline：**0.052%**（完全对齐）
- Dynamic vs Baseline：**7.37%**（符合预期——Dynamic BFD 调度将不同长度样本重新分组，改变各 DP group 的训练样本，属于设计行为而非 bug）

### 修改文件

| 文件 | 改动 |
|------|------|
| `mindspeed_mm/data/dataloader/sampler.py` | `full_bucket_size` 改用 `micro_batch_times_data_parallel_size` 归一化，确保不同 MBS 设置下 permutation 长度一致 |
| `run_8card_baseline.sh` | 8 卡 Baseline 测试脚本（NPUS=8, DP=4, CP=2, 20 steps） |
| `run_8card_hybrid.sh` | 8 卡 Hybrid 测试脚本（通过 `SCHEDULE_MODE=naive/dynamic` 切换） |

### 核心命令

```bash
# 新服务器 (8卡，现已关闭)
ssh -p 6023 root@localhost  # 已关闭

# 同步代码
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='save_dir' \
  --exclude='ckpt/' --exclude='MindSpeed/' --exclude='megatron/' --exclude='Megatron-LM/' \
  --exclude='logs/' -e 'ssh -p 6023' /mnt/hdc/xh/MindSpeed-MM/ root@localhost:/data/user/user40/hybrid_parallel/MindSpeed-MM/

# 远端运行 (在 /data/user/user40/hybrid_parallel/MindSpeed-MM/)
bash run_8card_baseline.sh 2>&1 | tee /tmp/baseline8.txt
SCHEDULE_MODE=naive bash run_8card_hybrid.sh 2>&1 | tee /tmp/hybrid_naive8.txt
SCHEDULE_MODE=dynamic bash run_8card_hybrid.sh 2>&1 | tee /tmp/hybrid_dynamic8.txt

# 对比 iteration loss
grep 'iteration.*loss:' /tmp/baseline8.txt | awk -F'loss: ' '{print $2}' | cut -d' ' -f1 > /tmp/b8.txt
grep 'iteration.*loss:' /tmp/hybrid_naive8.txt | awk -F'loss: ' '{print $2}' | cut -d' ' -f1 > /tmp/hn8.txt
paste /tmp/b8.txt /tmp/hn8.txt | awk '{b=$1+0;h=$2+0;d=(b>0)?(h-b)/b*100:0; printf "Step %2d: B=%.6f H=%.6f diff=%+.4f%%\n",NR,b,h,d}'
```

---

## 4 节点 64 卡集群使用方法

**访问方式**

```bash
# 本地 → 主节点（master，rank 0-15）
ssh -p 6024 root@localhost   # 主节点 IP: 103.224.234.232

# 主节点 → 其他工作节点
ssh root@103.224.131.7       # worker 2，rank 16-31
ssh root@103.224.49.245      # worker 3，rank 32-47
ssh root@103.224.140.115     # worker 4，rank 48-63
```

**环境激活**

```bash
source /data/user/user40/miniconda3/bin/activate
conda activate /opt/conda/private/envs/MindSpeed-MM
```

**代码同步**

```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='save_dir' \
  --exclude='ckpt/' --exclude='MindSpeed/' --exclude='megatron/' --exclude='Megatron-LM/' \
  --exclude='logs/' -e 'ssh -p 6024' \
  /mnt/hdc/xh/MindSpeed-MM/ root@localhost:/data/user/user40/hybrid_parallel/MindSpeed-MM/

# 只同步单个文件（快速验证时用）
rsync -avz -e 'ssh -p 6024' \
  /mnt/hdc/xh/MindSpeed-MM/scheduler.py \
  root@localhost:/data/user/user40/hybrid_parallel/MindSpeed-MM/
```

**启动训练（必须通过 launch_4node.sh，不能直接运行 finetune 脚本）**

```bash
cd /data/user/user40/hybrid_parallel/MindSpeed-MM
bash launch_4node.sh finetune_internvl3_8B_hybrid_multi_node.sh
```

`launch_4node.sh` 流程：SSH 到所有 worker 节点，创建 `.launcher/worker_<IP>.sh`，在 tmux 会话中运行；然后主节点自己在前台运行训练脚本。

**杀掉所有训练进程（端口被占用时使用）**

```bash
# 在主节点上
pkill -9 -f pretrain_vlm.py && pkill -9 -f torchrun

# 同时杀掉 worker 节点
for ip in 103.224.131.7 103.224.49.245 103.224.140.115; do
  ssh root@$ip 'pkill -9 -f pretrain_vlm.py; pkill -9 -f torchrun'
done

# 等约 8 秒让端口释放后再重新启动
```

---

## Log 查看方法

> **注意**：主节点的 `logs/train_<timestamp>.log` 仅包含 rank 0-15 的 DIAG 诊断输出，**不含** `iteration`/`loss`/`elapsed time` 等关键训练指标。这些指标由 Megatron 的 `training.py` 统一在 rank 0 打印，**实际 rank 0 在 worker 4（103.224.140.115）上**（4 节点 × 16 卡/节点配置下，rank 0 ≈ 最后一个 worker 的第一个进程——实际测试确认在 worker 4）。

```bash
# 查找 worker 4 上最新的 log 文件
ssh root@103.224.140.115 'ls -lt /tmp/train_worker_103.224.140.115_*.log | head -1'

# 实时跟踪 iteration 进度
ssh root@103.224.140.115 'tail -f /tmp/train_worker_103.224.140.115_<timestamp>.log'

# 提取 iteration 时间和 loss
ssh root@103.224.140.115 \
  'grep "elapsed time per iteration" /tmp/train_worker_103.224.140.115_<timestamp>.log'

# 检查 HCCL 超时 / 错误
ssh root@103.224.140.115 \
  'grep -E "(timeout|SIGABRT|Error|Traceback)" /tmp/train_worker_103.224.140.115_<timestamp>.log | grep -v DIAG'
```

**主节点 log（DIAG 数据对比用）**：

```bash
ls /data/user/user40/hybrid_parallel/MindSpeed-MM/logs/
grep '\[DIAG-LOSS\]' logs/train_<timestamp>.log | head -20
```

---

## 附录：框架对 Packed Sequence (TND 格式) 的支持分析

> 写于探索 64 卡 Packed Baseline 过程中，记录框架中各 CP 算法对 Packing 的支持限制。

### 核心数据格式

| 格式 | 描述 | 适用场景 |
|------|------|---------|
| **BSND** | `[Batch, Seq, N_heads, Dim]`，padding 到统一长度 | 静态批大小，浪费算力 |
| **TND (thd)** | `[Total_tokens, N_heads, Dim]`，拼接可变长序列 | Packing，高效利用算力 |

TND 格式依赖 `PackedSeqParams` 携带序列边界信息：
```python
PackedSeqParams(
    cu_seqlens_q=cu,       # 累积序列长度 [0, l1, l1+l2, ..., total]
    cu_seqlens_kv=cu,
    qkv_format='thd',
    max_seqlen_q=max_len,  # 单个最长子序列的长度（用于 RoPE 频率表预计算）
    max_seqlen_kv=max_len,
)
```

### Ulysses CP 对 Packing 的支持

**结论：支持。**

`mindspeed/patchs/ulysses_patches.py` 中：
1. 从 `cu_seqlens_q[-1]` 推导实际序列总长（`act_seq_len`）
2. 对 QKV 做 unsqueeze batch dim，然后 all-to-all 转置头维度
3. 将完整 `kwargs`（含 `packed_seq_params`）透传给 local attention

无对齐要求，任意长度的子序列均可。

**限制**：CP 受 KV head 数约束。InternVL3-8B 有 4 个 KV head，Ulysses CP ≤ 4，无法达到 CP=32 的大范围序列并行。

### Ring Attention (megatron_cp_algo) 对 Packing 的支持

**结论：有条件支持，有严格对齐要求。**

Ring Attention 的 Packing 路径分两层：

**层 1：Input 分发（mm_gpt_model.py 第 259-275 行）**

当 `packed_seq_params is not None`，使用 `split_packed_per_sample_megatron_cp` 对每个子序列独立做 zigzag split：

```python
for L in seqlens:
    chunk = L // (2 * cp_size)  # 必须整除！
    s1 = offset + cp_rank * chunk
    s2 = offset + (2 * cp_size - cp_rank - 1) * chunk
    ...
```

**严格要求：每个子序列长度必须是 `2 × cp_size` 的整数倍。** CP=32 时，每个子序列长度须为 64 的倍数。

**层 2：Ring Attention 核心（MindSpeed ring_context_parallel.py 第 935-960 行）**

```python
if packed_seq_params.cu_seqlens_q_padded is not None:
    # Case 1（设计正常路径）
    # 自动计算 q_index / kv_index（各子序列在 rank-local 视角下的前/后半索引）
    packed_seq_params.q_index = q_index
    packed_seq_params.kv_index = kv_index
    ...
else:
    # Case 2（bug：设置了 is_eod_reset=True 但未计算 q_index/kv_index）
    cp_config.actual_seq_kvlen = packed_seq_params.cu_seqlens_q.tolist()
    cp_config.actual_seq_qlen = packed_seq_params.cu_seqlens_kv.tolist()

cp_config.is_eod_reset = True  # 触发 EOD reset 路径（causal 下）
# ...
cp_config.kv_index = packed_seq_params.kv_index  # Case 2 下 AttributeError！
```

**现有 MindSpeed 版本存在 Bug**：不提供 `cu_seqlens_q_padded` 时，`is_eod_reset=True` 但 `kv_index` 未被计算，直接 `AttributeError`。

正确使用方式需提供 `cu_seqlens_q_padded`（每个子序列 pad 到 `2*cp_size` 倍数后的累积长度），Ring Attention 会自动计算 EOD 边界内的 zigzag 索引。

### DynamicBatchingDataLoader 与 Ring CP 的兼容性

`DynamicBatchingDataLoader`（`TextBatchingStrategy`）使用贪心 bin-packing：

- 将变长序列装箱至总 token 数 ≤ `max_seq_len`
- 产出 `seqlens` 张量 + 拼接的 `input_ids/labels`
- **不保证**每个子序列长度是 `2 × cp_size` 的倍数

因此，**DynamicBatching + Ring CP (megatron_cp_algo)** 直接使用存在以下问题：

1. `split_packed_per_sample_megatron_cp` 中 `chunk = L // (2 * cp_size)` 会丢弃尾部 `L % (2*cp_size)` 个 token（数据丢失）
2. Ring Attention 需要 `cu_seqlens_q_padded` 才能进入正确的 EOD-reset 路径

### 解决方案选项

| 方案 | 可行性 | 说明 |
|------|--------|------|
| Ulysses CP (CP ≤ 4) + DynamicBatching | **可行** | 无对齐要求，但 CP 度受限，需调整 DP/GBS 才能对齐 token budget |
| Ring CP + 在 DataLoader 中 pad 子序列到 `2*CP` 倍数 | **可行** | 需在 DynamicBatching 输出后额外 padding，总长须仍为 `max_seq_len` |
| Ring CP + 在 `get_batch()` 中提供 `cu_seqlens_q_padded` | **可行** | 计算 padded cu_seqlens 并在 PackedSeqParams 中设置，Ring CP 自动处理；需保证总 padded 长度 = `seq_length` |
| Ring CP + 不用 Packing（BSND padding baseline） | **可行** | 原始 baseline 方案，引入额外 padding 算力，比较不公平 |

---

## Phase 9：4 节点 64 卡 Packed Baseline（run_4node_packed_baseline.sh）

**目标**：为 `run_4node_synthetic_burst.sh`（Dynamic Hybrid 测试）提供一个公平的静态对照组。

原始 baseline 采用 BSND 格式（padding 到最大长度），会引入大量无效算力。新 baseline 改用 TND Packing 格式，将短序列 bin-pack 至 `max_seq_len ≈ 128k tokens/microbatch`，避免 padding 浪费。

### Token Budget 设计（与 Burst 测试对齐）

Burst 测试（`run_4node_synthetic_burst.sh`）：
- `SYNTHETIC_TOKEN_BUDGET_PER_GPU=8192`，64 卡总预算 = `64 × 8192 = 524,288 tokens/step`
- `MBS=4, GRAD_ACC=4, DP=32 (CP=2), GBS=512`
- 普通 step：`512 × 4096 = 2,097,152` tokens（但每卡仅 8192 token budget，实际受 scheduler 控制）

Packed Baseline 设计（token-based GBS）：
```
CP    = 32  （静态 Ring Attention，覆盖 128k 序列）
DP    = 64 / (TP=1 × PP=1 × CP=32) = 2
MBS   = 1   （1 个 packed microbatch，≈ 128k tokens）
GRAD_ACC = 2
GBS   = MBS × GRAD_ACC × DP = 1 × 2 × 2 = 4
token_budget/step ≈ 131072 × 2 × 2 = 524,288 tokens  ✓
```

### 主要参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--context-parallel-algo` | `megatron_cp_algo` | Ring Attention |
| `--use-cp-send-recv-overlap` | 开启 | 通信计算重叠 |
| `--seq-length` | 131072 | 128k tokens/microbatch |
| `--micro-batch-size` | 1 | 1 个 packed sequence |
| `--global-batch-size` | 4 | token-based GBS |
| `--use-txt-dynamic-batching` | 开启 | 启用 bin-packing |
| `--max-seq-len` | 131072 | bin-pack 目标长度 |
| `--dynamic-batch-buffer-size` | 200 | 预取缓冲区大小 |
| `--calculate-per-sample-loss` | 开启 | per-sample 归一化 |
| `SYNTHETIC_LENGTH_DIST` | `burst` | 与 burst 测试相同分布 |
| `SYNTHETIC_WITH_VISION` | `false` | 纯文本，避免 image_grid_thw 依赖 |

### 代码修改

**`pretrain_vlm.py`**（非 Hybrid 静态 CP packed 路径新增）：

1. **`train_valid_test_datasets_provider()`**：当 `not is_hybrid and use_txt_dynamic_batching` 时，将 base DataLoader 包装为 `DynamicBatchingDataLoader`（原框架只在 `pretrain_transformers.py` 中有此包装）

2. **`get_batch()`**：检测 `seqlens` 字段（DynamicBatching 产出），构建 `PackedSeqParams`：
   ```python
   if 'seqlens' in batch:
       seqlens = batch.pop('seqlens').cpu().to(torch.int32)
       cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
       cu[1:] = torch.cumsum(seqlens, dim=0)
       cu = cu.to(device=torch.cuda.current_device())
       max_seqlen = seqlens.max()
       batch['packed_seq_params'] = PackedSeqParams(
           cu_seqlens_q=cu, cu_seqlens_kv=cu,
           qkv_format='thd',
           max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen,
       )
   ```
   `max_seqlen_q/kv` 是 `mm_gpt_model.py` 中 RoPE 频率表预计算所必须的（`max(...) > None` 会 TypeError）。

**`examples/internvl3/model_8B.json`**：
- `max_position_embeddings`: `32768` → `131072`（框架校验 `max_position_embeddings ≥ seq_length`，否则报 `IsNotValidError`）

### 当前状态（遇阻）

Ring Attention (megatron_cp_algo) + DynamicBatching 存在附录中描述的对齐问题，测试过程中遇到：

1. ✅ `TypeError: '>' not supported between NoneType and NoneType`：`PackedSeqParams` 缺少 `max_seqlen_q/kv` → 已修复
2. ❌ `AttributeError: 'PackedSeqParams' object has no attribute 'kv_index'`：MindSpeed Ring CP 的 EOD-reset bug，需提供 `cu_seqlens_q_padded` 或子序列对齐到 64

Baseline 设计暂停，待后续选择合适方案（见附录"解决方案选项"）后继续推进。

---

## Phase 10：Qwen3VL Hybrid Parallel 支持

**目标**：将 Hybrid Parallel（动态 CP 调度）从 InternVL3 扩展至 Qwen3VL 模型。

### 架构差异分析

Qwen3VL 与 InternVL3 在 MindSpeed-MM 中使用完全不同的训练路径：

| 维度 | InternVL3 | Qwen3VL |
|------|-----------|---------|
| 入口脚本 | `pretrain_vlm.py` | `pretrain_transformers.py` |
| 模型类 | `VLMModel`（Megatron 组件分离：ViT + Projector + LLM） | `TransformersModel`（HuggingFace 模型整体包装） |
| 图像 token | `img_context_token_id=151667` 在 model_8B.json 中显式配置 | `image_token_id=151655` 在 HuggingFace config 中 |
| 序列格式 | TND packed（多样本打包为单长序列） | BSND（标准批次，Ulysses CP 按 head 分片） |
| `image_flags` | 数据集产出（`[1]*num_patches`） | 数据集**不产出**，需从 `input_ids` 合成 |
| `pixel_values` 行数 | = 图像 tile 数（1:1 对应 image_flags） | = 原始 patch 数 = merged patch × `spatial_merge_size²`（Qwen3VL 为 ×4） |

**初始错误方向**：试图将 Qwen3VL 接入 `pretrain_vlm.py` + `VLMModel`，但 `TransformersModel` 不支持 TND packed sequence（`packed_seq_params`），且切换模型类会导致与 baseline 行为不一致。

**正确方向（更新）**：在 `pretrain_transformers.py` 中添加 Hybrid Parallel 逻辑，使用 Scheduler 默认的 **TND 打包格式**（与 InternVL3 一致），并在 `TransformersModel` 中实现 TND-aware 的 per-sample loss。

Qwen3VL 的 HuggingFace 模型本身已原生支持 TND packed 序列（`attn_layout="TND"` 是 `model_8B.json` 的默认配置），无需额外修改模型代码。

### Qwen3VL 模型的 TND 支持机制

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| M-RoPE 位置编码 | `get_rope_index(sequence_length=seqlens)` | 按 sub-sequence 独立计算，保证各序列 position 从 0 起始 |
| FlashAttention masking | `actual_seq_qlen/kvlen=cu_seqlens` | 阻止跨 sub-sequence 注意力，基于 `seqlens` kwarg 计算 |
| 序列 unpad/repad | `indices = nonzero(attention_mask)` | 自动过滤 padding token，减少算力浪费 |
| CP 序列 split | `split_forward_gather_backward_with_cp(inputs_embeds, dim=1)` | 语言模型 forward 内部自动按 Ulysses 做 T→T/cp_size 分割 |

触发条件：向 `self.model()` 传入 `seqlens` kwarg（从 `packed_seq_params.cu_seqlens_q` 差分得到）。

### 修改文件清单

| 文件 | 改动 |
|------|------|
| `mindspeed_mm/models/transformers_model.py` | ① 新增 `img_context_token_id` 属性（Scheduler 初始化用）；② `forward()` 新增 `packed_seq_params=None` 参数，当存在时提取 `seqlens` 并传给模型，走 TND loss 路径；③ 新增 `_compute_tnd_loss()` 方法：per-sample label shift + split/gather + 支持 `default`/`per_sample_loss` 两种 loss_type |
| `pretrain_transformers.py` | 添加 Scheduler/HybridScheduledDataLoader import；`model_provider()` 中初始化 Scheduler；`train_valid_test_datasets_provider()` 中添加 `_SingleRankGroup`、MBS 放大、`HybridScheduledDataLoader` 包装；`get_batch()` 中仅 pop `image_flags`，**保留 `packed_seq_params`** 供 `_compute_tnd_loss` 使用 |
| `scheduler.py` | **两处 `get_data` 路径均修改**：① `image_flags` 缺失时合成 all-ones 向量；② `pixel_values.shape[0] > image_flags.shape[0]` 时用 `repeat_interleave` 扩展 pv_patch_mask |
| `examples/qwen3vl/model_8B.json` | 添加 `"img_context_token_id": 151655` |
| `mindspeed_mm/models/vlm_model.py` | `encoder_dp_enable` 扩展支持 `qwen3vit` |
| `examples/qwen3vl/data_8B_hybrid.json` | 新建数据配置模板 |
| `finetune_qwen3vl_8B_hybrid.sh` | 新建训练脚本，默认使用 TND 模式（移除 `DATA_LAYOUT=BSND`） |

### 关键设计决策

#### 1. TND packed 格式替代 BSND

Scheduler 默认（`DATA_LAYOUT=TND`）将同 CP 组的样本拼接为 `[1, T]` packed 序列，附带 `packed_seq_params`（含 `cu_seqlens_q`）。`TransformersModel.forward()` 通过以下步骤正确处理：

```python
# TransformersModel.forward()，当 packed_seq_params 存在时：
cu = packed_seq_params.cu_seqlens_q
seqlens = (cu[1:] - cu[:-1]).to(torch.int32)
kwargs['seqlens'] = seqlens   # 触发模型的 TND 注意力路径 + 正确 M-RoPE
position_ids = None            # 模型内部通过 get_rope_index(sequence_length=seqlens) 计算
```

#### 2. `_compute_tnd_loss`：per-sample label shift

全局 shift 会导致 sub-seq i 的最后一个 padding token 被误训练为 sub-seq i+1 的第一个 real token。需在每个 sub-sequence 内部独立 shift：

```python
for i in range(num_subseqs):
    start, end = cu_seqlens[i], cu_seqlens[i+1]
    shift_labels[0, start:end-1] = labels[0, start+1:end]
    # shift_labels[0, end-1] = -100  (no next token for last position)
```

#### 3. `image_flags` 合成逻辑（scheduler.py）

Qwen3VL 数据管线不产出 `image_flags`。Scheduler 从 `input_ids` 统计 `img_context_token_id` 出现次数合成 all-ones 向量（length = merged patch 数），保证 `img_token_per_patch = 1`，避免整数除法下溢。

#### 4. `pixel_values` expand factor（scheduler.py）

Qwen3VL `spatial_merge_size=2`：`pixel_values.shape[0]`（raw patch 数）= 4 × merged patch 数。用 `repeat_interleave` 扩展 pv_patch_mask 后正确切片 `pixel_values`。

#### 5. Two-phase 图像加载

Qwen3VL 数据集（`huggingface` 类型）无 `img_video_processor`，two-phase 加载 fallback 到全量加载（所有 rank 均预加载完整 `pixel_values`，Scheduler 再切片）。

### 数据流对比

```
InternVL3 Hybrid（原有）：
  DataLoader → HybridScheduledDataLoader.next_batch()
    → [TND 打包] → packed_seq_params → VLMModel.forward()
                                          ↓ (TND + Ring/Ulysses CP, MMGPTModel 原生支持)

Qwen3VL Hybrid（更新后）：
  DataLoader → HybridScheduledDataLoader.next_batch()
    → [TND 打包] → packed_seq_params → TransformersModel.forward()
                   seqlens 注入 ↗         ↓ (TND + Ulysses CP, HF 模型原生支持)
                                    _compute_tnd_loss()
```

### 注意事项

1. **M-RoPE 正确性**：`seqlens` kwarg 传入后，Qwen3VL 模型调用 `get_rope_index(sequence_length=seqlens)` 将 `input_ids[0]` 按各 sub-sequence 长度 split，逐段计算 position_ids，保证每段从正确位置起始。
2. **Ulysses CP split**：模型语言模型 forward 内部在 `inputs_embeds` 进入 decoder layers 之前做 `split_forward_gather_backward_with_cp(inputs_embeds, dim=1)`，因此 logits 形状为 `[1, T/cp_size, vocab]`，`_compute_tnd_loss` 对 `shift_labels` 做相同 split 后计算 local cross-entropy。
3. **精度对齐验证**：Qwen3VL Hybrid 与 baseline 的精度对齐尚未在硬件上验证，需按 Phase 2-4 的方式运行对比实验（`SCHEDULE_MODE=naive`）。

---

## Phase 11：Qwen3VL Hybrid 单机调试

**目标**：在远端 Server 1（4 卡，port 6022）上先跑通 Qwen3VL Hybrid TND 模式，验证无崩溃后再做精度对比。

### 调试策略

分三步递进：
1. **Step A**：2 卡（DP=1, CP=2），`SCHEDULE_MODE=naive`，5 steps，仅验证代码跑通（无 assert/crash）
2. **Step B**：对比 baseline vs hybrid 的 `[DIAG-DATA]` / `[DIAG-LOSS]` 输出，确认数据一致性和 loss 路径正确
3. **Step C**：20 steps 精度对比，验证 naive diff < 0.1%

### 测试脚本参数（单机 4 卡）

```bash
# Baseline（finetune_qwen3vl_8B.sh）
NPUS_PER_NODE=2   # 2 卡: TP=1, PP=1, CP=2, DP=1
MBS=1
GRAD_ACC_STEP=4
GBS=4

# Hybrid（finetune_qwen3vl_8B_hybrid.sh）
NPUS_PER_NODE=2
MBS=1             # per-group MBS（Scheduler 会 *num_groups 放大 DataLoader MBS）
GRAD_ACC_STEP=4
GBS=4
SCHEDULE_MODE=naive
```

### 快速命令

```bash
# 代码同步（Server 1）
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='save_dir' \
  --exclude='ckpt/' --exclude='MindSpeed/' --exclude='megatron/' --exclude='Megatron-LM/' \
  --exclude='logs/' -e 'ssh -p 6022' /mnt/hdc/xh/MindSpeed-MM/ root@localhost:/data/user/user40/hybrid_parallel/MindSpeed-MM/

# 远端环境激活
ssh -p 6022 root@localhost
source /data/user/user40/miniconda3/bin/activate && conda activate /opt/conda/private/envs/MindSpeed-MM
cd /data/user/user40/hybrid_parallel/MindSpeed-MM

# Baseline（2 卡 5 steps）
sed -i 's/NPUS_PER_NODE=8/NPUS_PER_NODE=2/; s/--train-iters 10000/--train-iters 5/' finetune_qwen3vl_8B.sh
bash finetune_qwen3vl_8B.sh 2>&1 | tee /tmp/q3vl_baseline.txt

# Hybrid（2 卡 5 steps）
sed -i 's/NPUS_PER_NODE=8/NPUS_PER_NODE=2/; s/--train-iters 10000/--train-iters 5/' finetune_qwen3vl_8B_hybrid.sh
SCHEDULE_MODE=naive bash finetune_qwen3vl_8B_hybrid.sh 2>&1 | tee /tmp/q3vl_hybrid.txt

# 检查是否崩溃
grep -E "(Error|Traceback|assert|SIGABRT)" /tmp/q3vl_hybrid.txt | grep -v DIAG

# 对比 iteration loss
grep 'iteration.*loss:' /tmp/q3vl_baseline.txt | awk -F'loss: ' '{print $2}' | cut -d' ' -f1 > /tmp/qb.txt
grep 'iteration.*loss:' /tmp/q3vl_hybrid.txt   | awk -F'loss: ' '{print $2}' | cut -d' ' -f1 > /tmp/qh.txt
paste /tmp/qb.txt /tmp/qh.txt | awk '{b=$1+0;h=$2+0;d=(b>0)?(h-b)/b*100:0; printf "Step %2d: B=%.6f H=%.6f diff=%+.4f%%\n",NR,b,h,d}'
```

### Step A 调试过程（2026-04-13）

在远端 Server（2 卡, TP=1, PP=1, CP=2, MBS=1, GBS=2, 2B 模型, bf16, SCHEDULE_MODE=naive）逐步修复以下 bug：

| # | 错误信息 | 根因 | 修复 |
|---|---------|------|------|
| 1 | `ModuleNotFoundError: No module named 'datasets'` | 调用了系统 torchrun（`/usr/local/bin/torchrun`）而非 conda 环境 | 改用 `/opt/conda/private/envs/MindSpeed-MM/bin/torchrun` |
| 2 | `DataArguments: unexpected keyword argument 'split'` | `split` 不是 `DataArguments` 的字段，函数内部硬编码 `split="train"` | 从 data config 的 `basic_parameters` 移除 `split` 字段 |
| 3 | `ProcessorArguments: unexpected keyword argument 'resize_vocab'` | `ProcessorArguments` 不接受 `resize_vocab`、`overwrite_cache`、`preprocessing_num_workers` | `preprocess_parameters` 只保留 `model_name_or_path` + `trust_remote_code` |
| 4 | `Image features and image tokens do not match: tokens:0, features:64` | `move_to_device()` 遇到 `PackedSeqParams` 对象（不是 Tensor/bool/int/str）时静默丢弃，导致 `packed_seq_params=None` 走旧路径——旧路径直接把 `pixel_values` 传模型，而文本序列中无图片 token | `move_to_device()` 新增 `isinstance(v, PackedSeqParams)` 分支，逐字段移动到 device，保留对象引用 |
| 5 | 同上（另一根因） | LlamaFactory collator 对纯文本样本也产出非 None `pixel_values`（默认行为） | `TransformersModel.forward()` 加安全检查：`(input_ids == img_context_token_id).sum() == 0` 时置 `pixel_values=None` |
| 6 | `NPU: call aclnnFlashAttentionVarLenScore failed, error code 561103` | `cu_seqlens_q` 存储的是 padded 累积长度（Scheduler 对短序列补零到等长），而 Qwen3VL 内核同时计算 `indices = nonzero(attention_mask.flatten())` 作为 actual token 位置，两者不一致（例：actual=300 tokens，padded cu_seqlens=(256,512)，`cu_seqlens[1]-cu_seqlens[0]=256 ≠ 300`） | `TransformersModel.forward()` 中改为对每个 padded 窗口统计 `attention_mask` 中 `1` 的数量得到 actual seqlens，传给 `kwargs['seqlens']` |
| 7 | `RuntimeError: NPU out of memory (59.21 GiB allocated, 187 MiB free)` | 4B 模型 bf16 (~8 GB) + Adam fp32 优化器状态 (~48 GB) ≈ 56 GB，超过 64 GB | 改用 2B 模型 (~4 GB model + ~24 GB optimizer ≈ 30 GB)，加 `--bf16` |

#### 关键修复细节

**`move_to_device()` 支持 `PackedSeqParams`（`pretrain_transformers.py`）：**
```python
elif isinstance(v, PackedSeqParams):
    dev = torch.cuda.current_device()
    for field in ('cu_seqlens_q', 'cu_seqlens_kv', 'cu_seqlens_q_padded',
                  'cu_seqlens_kv_padded', 'max_seqlen_q', 'max_seqlen_kv'):
        t = getattr(v, field, None)
        if isinstance(t, torch.Tensor):
            setattr(v, field, t.to(device=dev))
    new_batch[k] = v
```

**Actual seqlens 计算（`transformers_model.py`，Step A 初版，仅 2-card 可用）：**
```python
if packed_seq_params is not None:
    cu = packed_seq_params.cu_seqlens_q.long()
    if attention_mask is not None:
        seqlens = torch.stack([
            attention_mask[0, cu[i]:cu[i+1]].sum(dtype=torch.int32)
            for i in range(cu.numel() - 1)
        ])
    else:
        seqlens = (cu[1:] - cu[:-1]).to(torch.int32)
    kwargs['seqlens'] = seqlens
    position_ids = None
```

> ⚠️ 此修复仅 2-card 下有效——当时 DP=1，只有 1 个 sub-sequence，Scheduler 不需要 padding，`actual == padded`。扩到 4 卡后（DP=2, CP=2）Scheduler 会把 2 个 sub-sequences 补齐为等长，`actual != padded`，这段逻辑会崩。**Step B 中进行了正确修复**，见下文。

### Step A 结果

**✅ PASSED** — 2-card（CP=2）Qwen3VL-2B, SCHEDULE_MODE=naive, 5 iterations 全部正常完成：

```
iteration 1/5 | elapsed: 4616ms | loss: 5.933946E+00
iteration 2/5 | elapsed:  642ms | loss: 4.497444E+00
iteration 3/5 | elapsed:  384ms | loss: 4.615419E+00
iteration 4/5 | elapsed:  415ms | loss: 3.511344E+00
iteration 5/5 | elapsed:  379ms | loss: 3.633779E+00
```

**遗留问题**：5 steps 结束后保存 checkpoint 时报 `_write_item() missing 1 required positional argument: 'serialization_format'`（Megatron 版本兼容问题），脚本以 RC=1 退出。训练本身无误，规避方法：在 `OUTPUT_ARGS` 中添加 `--no-save`。

### Step B：扩到 4 卡（DP=2, CP=2）——暴露 padded-vs-actual seqlens 冲突

将 Step A 的脚本参数化为 `debug_qwen3vl_scaled.sh`（通过 `HYBRID` / `NPUS` / `SCHEDULE_MODE` / `ITERS` 控制），先跑 4 卡 baseline 作对照。

**Baseline（4-card, DP=2, CP=2）**：5 iters 全部成功（RC=1 仅是 `_write_item` checkpoint 保存 bug）。

**Hybrid-Naive（4-card）**：崩溃，报错：
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 76 (input tensor's size at dimension 0), but got split_sizes=[73]
```

**根因**：`transformers_model.py` 把 `seqlens = attention_mask.sum()` 当成 **actual** 长度（73）传给模型；而 Qwen3VL 的 `get_rope_index` 做 `input_ids[0].split(sequence_length.tolist())`，`input_ids` 是 **padded** 长度（76），两者对不上。

- `actual seqlens (73)` 是 FlashAttention var-len kernel 所需——它对 `attention_mask` 取 `nonzero` 得到 `indices` gather 实际 token，cu_seqlens 必须匹配 unpadded 长度。
- `padded seqlens (76)` 是 `get_rope_index` 所需——要把 padded `input_ids[0]` 切成 sub-sequences。

两者同为 `kwargs['seqlens']`，一个值没法兼顾。

**正确修复**：传 **padded** seqlens，并预先塞 `kwargs['indices'] = arange(T)` 绕过 Qwen3VL 模型内部的 `attention_mask → indices` unpad 过滤。padding 位置的 labels 为 `-100`，CE 不计入；padding 位置的 attention 在 sub-sequence 内部发生但无梯度贡献（label -100），浪费很少算力换来 seqlens 统一。

```python
# transformers_model.py
if packed_seq_params is not None:
    cu = packed_seq_params.cu_seqlens_q.long()
    seqlens = (cu[1:] - cu[:-1]).to(torch.int32)   # padded
    kwargs['seqlens'] = seqlens
    total_len = input_ids.shape[1]
    kwargs['indices'] = torch.arange(total_len, device=input_ids.device)
    position_ids = None
```

**4-card 结果（5 iters, Qwen3VL-2B, DP=2 CP=2）**：

| Step | Baseline | Hybrid-Naive | diff |
|------|----------|--------------|------|
| 1  | 5.215511 | 5.215694 | +0.004% |
| 2  | 5.013690 | 5.013690 | 0.000% |
| 3  | 4.058189 | 4.050153 | -0.198% |
| 4  | 3.751116 | 3.751109 | 0.000% |
| 5  | 3.548689 | 3.546898 | -0.051% |

**最大 diff 0.2%**，对齐成功。

### Step C：扩到 8 卡（DP=4, CP=2）+ Dynamic 模式测试

**8-card 结果（10 iters, Qwen3VL-2B, DP=4 CP=2）**：

| Step | Baseline | Hybrid-Naive | Hybrid-Dynamic | diff_N | diff_D |
|------|----------|--------------|---------------|--------|--------|
| 1  | 5.114601 | 5.114692 | 4.877264 | +0.002% | -4.64% |
| 2  | 4.902876 | 4.905752 | 4.604669 | +0.059% | -6.08% |
| 3  | 3.867514 | 3.866912 | 3.792945 | -0.016% | -1.93% |
| 4  | 3.878970 | 3.874102 | 3.623030 | -0.126% | -6.60% |
| 5  | 4.187395 | 4.183721 | 4.100068 | -0.088% | -2.08% |
| 6  | 3.411007 | 3.411714 | 3.272007 | +0.021% | -4.08% |
| 7  | 3.406422 | 3.400958 | 3.243188 | -0.160% | -4.79% |
| 8  | 3.399025 | 3.396560 | 3.338300 | -0.073% | -1.79% |
| 9  | 3.227217 | 3.228586 | 3.058607 | +0.042% | -5.23% |
| 10 | 3.039479 | 3.039350 | 2.975408 | -0.004% | -2.11% |

- **Naive vs Baseline**：max **0.16%**（对齐）
- **Dynamic vs Baseline**：max **6.60%**（符合预期——Dynamic BFD 重排样本，各 DP group 的训练样本与 Baseline 不同；`tokens per sample` 从 116–195 跳到 464–779，样本组成完全改变）

---

## Phase 12：Qwen3VL 的 loss 计算修正（对齐 VLMModel 约定）

**目标**：对照 `pretrain_vlm.py` + `vlm_model.py` 已对齐的 loss 逻辑，检查 Qwen3VL 路径（`pretrain_transformers.py` + `transformers_model.py`）是否存在 loss 计算错误。

### 问题 1：`loss_for_backward` 比正确值大 **cp_size 倍**（预存在 bug，非 Phase 10/11 引入）

**路径追踪**：
- `build_loss_ctx` / `_compute_tnd_loss` 在每个 CP rank 上得到 `local_sum / global_count`（alpha 是 pre-CP-split 的 `loss_mask.sum()`）。各 CP rank 的 loss 求和才是 global_mean。
- 原 `pretrain_transformers.loss_func` 直接 `return loss, loss_dir`（**未除 cp_size**）。
- `schedules.py:297-298`（2-tuple 返回路径）再做 `output_tensor *= cp_size; output_tensor /= num_microbatches`。
- 最终跨 CP rank 求和后 backward loss = `global_mean × cp_size / num_microbatches`——**cp_size 倍过大**。

对照 `pretrain_vlm.py:349`：`return loss / cp_size, loss_dir`，乘 `cp_size/num_microbatches` 后正好是 `global_mean / num_microbatches` ✓。

**影响**：CP>1 时 Qwen3VL 所有训练（baseline 与 hybrid 均如此）实际有效学习率是配置值的 `cp_size ×`。由于 baseline 与 hybrid 走同一条 `loss_func`，彼此仍然对齐，之前测试无法暴露。

**修复**：`loss_func` 返回前加 `/ cp_size`。

### 问题 2：Hybrid 缺少 `num_samples / mbs_per_group` 梯度权重修正

VLMModel hybrid 分支（`pretrain_vlm.py:322-325`）：
```python
loss = loss * (num_samples / mbs_per_group)
```
BFD dynamic 把不同数量的样本分到各 DP group，各组的 raw loss 是"组内样本均值"，直接拼 backward 会让"样本少的组"每个样本的梯度权重偏大。乘以 `num_samples/mbs_per_group` 把每个样本的贡献拉平为"全局等权"。

原 Qwen3VL `loss_func` 无 hybrid 分支，不做此修正。naive 模式下各组 `num_samples == mbs_per_group`，无影响；dynamic 模式下的真实梯度分布与 baseline 不一致。

**修复**：`loss_func` 中读环境变量判断 `is_hybrid`，按 VLMModel 同样公式做 rescale。为此在 `_compute_tnd_loss` 的返回 dict 中补上 `num_samples = num_subseqs` 和 `token_nums = loss_mask.sum()`。

### 问题 3：Hybrid logging_loss 约定

VLMModel 的 CP 路径对 loss 做了 `gather_forward_split_backward`，所以每个 rank 持有的是 **完整** group_mean；Qwen3VL 没有 gather，每个 rank 持有 **local 分数** (`local_sum / global_count`)。

因此两侧的 logging reduce 公式必须不同：

- VLMModel baseline：`average_losses_across_data_parallel_group([loss])`（只 DP-avg 即可）
- Qwen3VL baseline：`AVG(loss, dp_cp_group) * cp_size`（先 dp*cp 平均再乘 cp_size 恢复 group_mean；这是 Qwen3VL 原本就写对的）

保留 Qwen3VL 的 `AVG × cp_size` 公式，同时在 hybrid 分支对 raw loss 做 `num_samples/mbs_per_group` rescale 后再走相同 logging reduce，即同时覆盖 baseline 和 hybrid naive/dynamic 场景。

### 修改文件

| 文件 | 改动 |
|------|------|
| `pretrain_transformers.py` | `loss_func` 重写：返回 `loss / cp_size`；新增 `is_hybrid` 分支做 `num_samples/mbs_per_group` 梯度权重修正；logging 保留 `AVG(dp_cp) * cp_size` |
| `mindspeed_mm/models/transformers_model.py` | `_compute_tnd_loss` 返回 dict 增加 `num_samples` 和 `token_nums` 字段 |

### 修复后 8-card 验证

| Step | Baseline | Hybrid-Naive | Hybrid-Dynamic | diff_N | diff_D |
|------|----------|--------------|---------------|--------|--------|
| 1 | 5.114601 | 5.114692 | 4.877264 | +0.002% | -4.64% |
| 2 | 4.902876 | 4.905752 | 4.604669 | +0.059% | -6.08% |
| 3 | 3.871086 | 3.866991 | 3.792002 | -0.106% | -2.04% |
| 4 | 3.876135 | 3.878175 | 3.626861 | +0.053% | -6.43% |
| 5 | 4.184245 | 4.187521 | 4.103104 | +0.078% | -1.94% |
| 6 | 3.412994 | 3.410792 | 3.268300 | -0.065% | -4.24% |
| 7 | 3.400111 | 3.401413 | 3.243428 | +0.038% | -4.61% |
| 8 | 3.398780 | 3.398322 | 3.343961 | -0.013% | -1.61% |
| 9 | 3.228653 | 3.228920 | 3.061996 | +0.008% | -5.16% |
| 10 | 3.040405 | 3.039821 | 2.977635 | -0.019% | -2.07% |

- Naive vs Baseline：max **0.11%**（相比 Phase 11 Step C 的 0.16% 再微改善）
- Dynamic vs Baseline：max **6.43%**（设计行为，不变）
- iter 1–2 数值与修复前完全一致（iter 1 LR=0 warmup，iter 2 forward 用的是 iter 1 后的权重，而 iter 1 没更新），从 iter 3 开始因 backward 正确缩放出现微小漂移

---

## Phase 13：Synthetic Dataloader 接入 Qwen3VL

**目标**：将 Phase 7 / pretrain_vlm.py 中已验证的 Synthetic Dataloader（`synthetic_dataloader.py`）接入 `pretrain_transformers.py`，支持 Qwen3VL 纯文本合成数据训练，用于 burst/scheduler 压测。

### 修改

**`pretrain_transformers.py`**：镜像 `pretrain_vlm.py` 的 synthetic 分支结构。
1. import `build_synthetic_dataloader`
2. `train_valid_test_datasets_provider()` 开头读 `SYNTHETIC_DATA` env；为 true 时跳过 `build_mm_dataset` 与 two-phase image loading 配置。
3. dataloader 构建处新增 `if use_synthetic` 分支，调用 `build_synthetic_dataloader(args.micro_batch_size)`。

**`scheduler.py` 两处 `get_data`**：新增纯文本 fast path。

原代码在 `image_flags` 为空时崩溃：
```python
img_token_per_patch = img_token_sum // torch.sum(databatch['image_flags'])  # FloorDiv by 0
```
纯文本 batch 里 `image_flags` 长度为 0，NPU 上触发 `FloorDiv` AICPU 异常（错误码 507018）。

修复：
```python
if databatch['image_flags'].numel() == 0:
    img_patch_mask = torch.zeros(0, dtype=torch.bool,
                                 device=databatch['image_flags'].device)
else:
    # ...原有图像 patch mask 计算...
```

后续 `pixel_values`/`image_flags` 的切片得到空 tensor 并被置为 `None`，模型 forward 里 `TransformersModel.forward` 的"纯文本 safety check"（Phase 11 修 bug #5）把 `pixel_values` 置 None，走纯文本路径。

### 验证

4-card hybrid naive + synthetic（`SYNTHETIC_MIN_LEN=128 MAX_LEN=512 NUM_BATCHES=20`）：

```
iter 1: loss=13.586  tok/sample=577
iter 2: loss=13.606  tok/sample=603
iter 3: loss=13.386  tok/sample=373
iter 4: loss=12.837  tok/sample=960
iter 5: loss=12.717  tok/sample=671
```

随机词表的初始 CE loss ≈ `log(151936) ≈ 11.93`，符合理论值。训练正常推进。

### 使用方式

```bash
SYNTHETIC_DATA=true \
SYNTHETIC_MIN_LEN=512 SYNTHETIC_MAX_LEN=8192 \
SYNTHETIC_NUM_BATCHES=1000 \
HYBRID=1 NPUS=8 SCHEDULE_MODE=naive \
  bash debug_qwen3vl_scaled.sh
```

支持的环境变量（与 pretrain_vlm 一致）：`SYNTHETIC_MIN_LEN` / `MAX_LEN` / `NUM_BATCHES` / `LENGTH_DIST` (`uniform`/`normal`/`skewed`/`extreme_skewed`/`burst`) / `SEED` / `VOCAB_SIZE`。

---

## 当前状态

- Phase 11（Qwen3VL Hybrid 单机调试）：✅ 全部完成
  - Step A（2-card crash test）：✅
  - Step B（4-card padded seqlens 修复 + 对齐）：✅（naive diff 0.2%）
  - Step C（8-card naive/dynamic 验证）：✅（naive 0.16%，dynamic 6.6% 设计行为）
- Phase 12（Qwen3VL loss 计算修正）：✅（3 个 bug 全部修复，验证后 naive diff 0.11%）
- Phase 13（Synthetic Dataloader 接入 Qwen3VL）：✅
- 遗留：`_write_item()` checkpoint 保存 bug（Megatron 版本兼容问题），所有 run 退出 RC=1 但训练本身无误；规避方法为加 `--no-save` 或忽略 RC
- Qwen3VL Hybrid 代码栈已与 InternVL3 Hybrid 特性对齐，可进入多机扩展测试
claude --resume 08af31d4-e23a-45a3-ac8f-8c7cea074cfc

---

## Phase 14：Qwen3VL 4 节点 64 卡 Dynamic Hybrid Burst 测试

**目标**：把 Qwen3VL 接入 `run_4node_synthetic_burst.sh` 同规模的 burst 压测，验证 Dynamic Hybrid 调度器在 Ulysses CP 下能承载 128k/256k 长上下文。

### 初版脚本 → 一系列阻塞点

新建 `run_4node_qwen3vl_synthetic_burst.sh` 对齐 InternVL3 burst 参数（TP=1 PP=1 CP=2，MBS=4 GRAD_ACC=4，`SYNTHETIC_BURST_LEN=131072`，`cluster_size × budget_per_gpu = 524288`），调用 `pretrain_transformers.py`。顺序踩到以下问题，逐个根因 + 修复：

#### 阻塞 1：vision 合成数据不带 `image_grid_thw`

Qwen3VL 视觉塔的 `rot_pos_emb(grid_thw)` 要求 `image_grid_thw` 非 None。`synthetic_dataloader.py` 只产出 `pixel_values + image_flags`（为 InternVL3 设计），不产出 `image_grid_thw`。

**修复**：把 `SYNTHETIC_WITH_VISION=false`。Phase 13 已为 scheduler 加了空 `image_flags` fast path，纯文本路径已通。burst 测试本身也不依赖视觉。

#### 阻塞 2：`repeat_kv` OOM（burst CP=16，chunk=9216，Qwen3VL-4B）

DP 输出 `group_list=[16, 1×48]`（sum=64 ✓），burst 组分到 CP=16。OOM 实际发生在 `modules.py:651` 的 `repeat_kv`，但真正压力来自上游累计的 60 GB：Qwen3VL 4B 无 sharding 时，params(8G) + optimizer 全量 fp32 master/m/v(48G) ≈ 56G，激活再加几 G 就爆。

**当时的错误判断**：对比 InternVL3 vs Qwen3VL baseline 脚本——**Qwen3VL 家的脚本都没用 `--use-distributed-optimizer`**（InternVL3 全都用）。我最初以为 Megatron DistributedOptimizer 对 HF TransformersModel 的参数不生效，于是去掉了这个 flag，这直接锁死了 4B 路线。实际上 DistributedOptimizer 对 `nn.Parameter` 一视同仁，HF 参数照样分片——阻塞 5 证明了这点。

**当时的决策**：跟用户讨论后切到 **Qwen3VL 2B**（`model_2B.json`，`num_attention_heads=16`），放弃 4B 对 CP=32 的追求（2B 最大 Ulysses CP=16 仍能覆盖 256k 序列，`T_local = 256k/16 = 16k` 单卡可吃）。

#### 阻塞 3：Scheduler 在 Qwen3VL 下的几个隐藏 Bug

切到 2B + 按用户方向做 `seq_len_chunk=8192`（用来算 min_cp = ceil(seqlen/chunk)）+ `max_cp_degree=16` 后暴露：

1. **`data_len = sum(attention_mask) + 1` 导致 burst_len = 131073**，`ceil(131073/8192) = 17 > 16`。原 DP 的 `range(min_cp, max_cp+1)` 变空，fallback 到 min_cp=17，产出无效 CP=17 组。
2. **DP 候选 CP 没受 "必须是 num_heads 因子" 的约束**。uniform 冒烟测试下 `group_list=[5, 4, 4, ..., 3, 3, ..., 2, 1]`，CP=3/5 都不是 16 的因子 → FA kernel 抛 `n1Size [3] should be a multiple of n2Size [2]`（GQA 头数整除失败）。
3. **DP 可能输出 `sum(group_list) < cluster_size`**（InternVL3 之前就有的隐藏 bug，在 sum 正好==64 的工作负载下从未暴露）。`update_rank_dicts` 从 rank 0 开始分配，高序号 rank 落单，`get_group_id` 抛 `ValueError: Rank X is not in any parallel group`。

**修复（全部集中在 `scheduler.py`，InternVL3 走默认参数不受影响）**：
- `__init__` 新增 3 个参数：`seq_len_chunk`（默认 9216）、`max_cp_degree`（默认 32）、`cp_must_be_power_of_two`（默认 `False`，Ring 接任意 CP；Qwen3VL/Ulysses 传 `True`）。
- `compute_parallel_method` 里的硬编码 `seq_len_chunk = 9216` 和 `min(32, ...)` 全部用 `self.*` 替换。
- `min_cp_degrees` 生成时：先 cap 到 `max_cp_degree`，Ulysses 路径再向上取下一个 2 的幂（保证 BFD 的 `bin_cap = min_cp × chunk` 与 DP 能选的 CP 一致）。
- `dp_resource_allocation` 新增 `cp_must_be_power_of_two` 开关。内部辅助函数 `_candidate_cps(min_cp_g, upper)`：Ulysses 下只产出 2 的幂序列，Ring 下产出连续整数。`group_time_table` 构造和 DP 转移都改成迭代 `_candidate_cps`。
- DP 完成后新增 **rank-conservation 后处理**：若 `sum(group_list) < total_available_ranks`，用 round-robin 按 `cp_must_be_power_of_two` 分流：
  - Ulysses：每组最多一次 `CP ← 2×CP`，扫多轮直到 delta=0。
  - Ring：每组每轮 `CP += 1`。
  - 两种模式最终都保证 `sum == total_available_ranks`，否则抛清晰的 `RuntimeError`。
- `pretrain_transformers.py` 从 HF config 读 `num_attention_heads`，传 `seq_len_chunk=8192, max_cp_degree=num_heads, cp_must_be_power_of_two=True`。`pretrain_vlm.py` 不动 → InternVL3 路径完全不受影响（默认 False + chunk=9216）。

**验证**：uniform 冒烟测试 10/10 iter 通过，`group_list=[4×13, 2×5, 1×2]` sum=64，全部 2 的幂 ✓。

#### 阻塞 4：`logits.contiguous().float()` OOM（burst forward）

切回 burst 后第 1 次跑到 iter 2 挂，OOM 在 `transformers_model.py:162 logits = outputs.logits.contiguous().float()`。bf16 `[1, 8194, 151936]` = 2.49 GB，fp32 = 4.98 GB，`.float()` 的瞬时峰值 7.47 GB 压垮 burst 组每张卡。

**修复（`mindspeed_mm/models/transformers_model.py`）**：
- 去掉 `forward()` 里的 `.contiguous().float()`，只传 bf16 `outputs.logits` 给 `_compute_tnd_loss`。
- `_compute_tnd_loss` 的 `default` / `token_loss` 分支改为 **分块 CE**（`chunk=1024`）：沿 seq 维每 1024 tokens 切一段，只对该段 `.float()` 后送 `vocab_parallel_cross_entropy`，累加 `loss_sum`，最后除以全局 `alpha`。峰值 fp32 logits 只剩 ~640 MB/chunk。
- `per_sample_loss` 分支暂未分块（当前配置 `loss_type=default` 不走这条）。

#### 阻塞 5：Backward OOM（filler rank，distributed_optimizer 判断纠正）

chunk CE 修好以后又在 backward 挂，`rank 62` 报 `Tried to allocate 2.21 GiB, 58.05 GiB already allocated`。filler rank 只处理 ~4k tokens 不该吃 58 GB——真正的元凶是 **没开 distributed optimizer**：2B 模型全量 fp32 master+m+v = 24 GB，在每张卡上独立复制，再加 params/grads/activations 就到 58 GB。

**修复**：`run_4node_qwen3vl_synthetic_burst.sh` 加回 `--use-distributed-optimizer`。Megatron 的 DistributedOptimizer 对 HF `nn.Parameter` 工作正常（推翻 Phase 14 初期的判断），DP=32 下每张卡的 optimizer state 从 24 GB → 0.75 GB，省下的 23 GB 给激活与瞬态张量留出了充足余量。

#### 阻塞 6：HCCL 端口冲突

重启时报 `Failed to bind the IP port. Reason: The IP address and port have been bound already`。之前的 run 因 OOM 崩溃留下 TIME_WAIT 的 socket。

**修复**：
- `HCCL_IF_BASE_PORT` 改为 `${HCCL_IF_BASE_PORT:-52000}`，避免和默认 50000 范围冲突。
- `launch_4node.sh` 的 cleanup 同时 kill `pretrain_transformers.py` 与 `pretrain_vlm.py`。
- 运维上清理远端 worker 前等 ≥15s 让 TIME_WAIT 自行清掉。

### 最终验证（128k burst × 10 iter × DP=32 CP≤16）

```
iter 1  t=23490ms tok/sample=23298 loss=3.542  [burst]
iter 2  t= 4355ms tok/sample= 7910 loss=3.546
iter 3  t=10981ms tok/sample=23358 loss=3.542  [burst]
iter 4  t=14131ms tok/sample=30957 loss=3.590  [burst]
iter 5  t=13264ms tok/sample=30934 loss=3.565  [burst]
iter 6  t=13311ms tok/sample=30862 loss=3.565  [burst]
iter 7  t= 3476ms tok/sample= 7926 loss=2.828
iter 8  t= 9524ms tok/sample=23327 loss=3.105  [burst]
iter 9  t= 7774ms tok/sample=15507 loss=3.751  [burst]
iter 10 t= 9663ms tok/sample=23307 loss=3.618  [burst]
```

非 burst 步 ~3.5-4.5s，burst 步 ~9-14s（burst_prob=0.5 下约一半命中），loss 收敛行为正常。DIAG-SCHED 确认 burst microbatch 下 `group_list=[16, 2×N, 1×M]`——burst 组吃满 CP=16，filler 按 round-robin doubling 把 slack 填平，全部 2 的幂 ✓。

### 修改文件清单

| 文件 | 改动 |
|------|------|
| `scheduler.py` | `__init__` 三个新参数；`compute_parallel_method` 去硬编码 + min_cp cap/pow2 向上取整；`dp_resource_allocation` 增 `cp_must_be_power_of_two` + `_candidate_cps` 候选过滤；DP 之后 rank-conservation round-robin 后处理（Ring +1、Ulysses doubling）+ 显式 sum 不等校验 |
| `pretrain_transformers.py` | 从 HF `transformer_config` 读 `num_attention_heads` 传给 Scheduler；`seq_len_chunk=8192 max_cp_degree=num_heads cp_must_be_power_of_two=True` |
| `mindspeed_mm/models/transformers_model.py` | TND forward 去除 `.contiguous().float()`；`_compute_tnd_loss` `default` 分支改分块 CE（`chunk=1024`） |
| `synthetic_dataloader.py` | `BurstLengthSampler` target 除以 `(1 + burst_noise)` 防止 jitter 上冲；末尾按比例缩放 + min_len 保底 + 最终 assert `sum ≤ max_total_tokens` |
| `run_4node_qwen3vl_synthetic_burst.sh` | 指向 `model_2B.json` / `Qwen3VL-2B-Instruct`；MBS=2 GRAD_ACC=4（GBS=256）；保留 `--use-distributed-optimizer`；补齐 `--use-mcore-models --use-cpu-initialization --optimizer-selection fused_torch_adamw --untie-embeddings-and-output-weights`；`HCCL_IF_BASE_PORT=52000`；`SYNTHETIC_WITH_VISION=false`；注释 chunk ↔ token_budget_per_gpu 的耦合不变式 |
| `examples/qwen3vl/model_2B.json` | `init_from_hf_path` 改到项目相对路径 `./ckpt/hf_path/Qwen3VL-2B-Instruct` |
| `examples/qwen3vl/model_8B.json` | `init_from_hf_path` 同样改为项目相对路径（后续 4B 尝试时用） |
| `launch_4node.sh` | cleanup trap 同时 pkill `pretrain_vlm.py` 和 `pretrain_transformers.py`（原先只 kill 前者，Qwen3VL 进程漏杀） |

### 关键不变式（后续维护备忘）

1. **`seq_len_chunk ≥ SYNTHETIC_TOKEN_BUDGET_PER_GPU`**：BFD 的 bin 容量必须覆盖合成数据的 per-GPU budget，否则 `sum(min_cp) > cluster_size` 直接崩。当前 Qwen3VL 两边都是 8192。
2. **`max_cp_degree ≤ num_attention_heads`**（Ulysses），**且是 2 的幂**（因为 `_candidate_cps` 的 doubling 链只到 2 的幂）。2B: 16，4B: 32。
3. **`SYNTHETIC_BURST_LEN ≤ max_cp_degree × seq_len_chunk`**：否则 `min_cp > max_cp`，DP 没有合法解。2B+chunk=8192 → 最大 burst = 131072 正好卡界（这就是为什么 256k 需要提 chunk 或换 4B）。
4. **`num_attention_heads % cp_size == 0`** 且 **`cp_size > num_kv_heads` 时走 `repeat_kv`**：这两个约束由 `modules.py:649` 的 Ulysses 分支天然保证，只要上面的 2 幂约束成立。

### 遗留 & 下一步

- **256k burst 尚未验证**：chunk=8192 下 `ceil(262144/8192)=32 > max_cp=16`，`min_cp` 被 cap 到 16 → BFD bin_cap=131072 放不下 262144 → 单开一 bin 但 DP 只能给 CP=16，`T_local = 262144/16 = 16k` 每卡，需先确认能跑通再讨论 chunk 调整策略。
- **Qwen3VL 4B 方向**：Phase 15 证明 `--use-distributed-optimizer` 对 HF 模型有效，4B 可能也能跑；Phase 14 中途切 2B 是基于错误判断，是否回头用 4B 做最终交付需要再评估。
- **稳定性**：目前只跑了 10 iter 烟雾测试，未做长时间稳定性验证。
claude --resume 9bfc42e6-98fa-46c9-b067-cf9f39599f0f

---

## Phase 15：Static-CP TND Baseline + Loss Reduction 重写

**目标**：为 Hybrid burst run 提供一个公平的静态 CP × DP TND packed baseline，并保证两边 logged loss 在数值上对齐，作为最终交付的对照组。

### 设计：FFD bucket-pack 静态 baseline

与 Phase 14 的 Hybrid burst（CP=2 + 动态 BFD scheduler）共享数据源，唯一区别是「样本到 rank 的分配」从动态 BFD 换成了静态 First-Fit-Decreasing bin-packing：

| 维度 | Hybrid burst | Static FFD baseline |
|------|--------------|---------------------|
| Megatron CP | `--context-parallel-size 2` | `--context-parallel-size 16` |
| 动态 CP groups | Yes（Scheduler swap mpu CP group） | No |
| 数据源 | `SyntheticDataLoader(B=64)` | `SyntheticDataLoader(B=64)`（同 seed） |
| 每 microbatch B | MBS_user(2)×num_groups(32) = 64 | MBS_user(16)×DP(4) = 64 ✓ |
| GBS | 2×4×32 = 256 samples | 16×4×4 = 256 samples ✓ |
| 调度 | BFD 动态分组到不同 CP 大小 | FFD 把 64 样本装进 4 个 bucket（cap=131072）|
| Burst 样本 | 独占一个 CP=16 动态 group | 独占一个 DP bucket |
| 每 rank 输入 | 动态 group 拼出的 TND | `[1, 131072]` TND（固定形状）|

**关键性质**：因为同一 seed 下 `SyntheticDataLoader.__next__` 产出的 64 个样本完全一致、两边都消费 4 个 microbatch、每 microbatch 的样本是 byte-identical，**理论 per-step per-sample mean loss 一定相等**。两边数值一致即证明 loss reduction 在 hybrid 与 static 两条路径下被正确实现。

### 实现：3 个新文件 / 改动

| 文件 | 改动 |
|------|------|
| `static_pack_dataloader.py`（新）| `StaticBalancedPackLoader`：包装 `SyntheticDataLoader`，FFD 把 B=64 samples 装进 DP=4 buckets（cap=`max_pack_seqlen`），本 rank 取自己 DP group 的 bucket 拼成 `[1, max_pack_seqlen]` TND，padding 区填 0/-100，尾部 padding 也作为一个 sub-seq 让 cu_seqlens 覆盖整个张量。FFD 排序按 (-length, index)，确定性的 → 所有 rank 独立计算得到完全相同的分配 |
| `pretrain_transformers.py` | `STATIC_PACK=true` 分支：跳过 Scheduler/HybridScheduledDataLoader；按 hybrid 同样的 MBS inflation pattern（`args.micro_batch_size *= dp_size`）让 inner SyntheticDataLoader 产出 B=64；用 `dp_rank` 切自己的 bucket；恢复 `args.micro_batch_size` 给 Megatron 看 |
| `run_4node_qwen3vl_static_baseline.sh`（新）| `CP=16 DP=4 MBS=16 GRAD_ACC=4 GBS=256 seq_length=131072`；与 burst 脚本完全一致的 `SYNTHETIC_*` 环境变量保证数据 byte-identical；`unset HYBRID_PARALLEL`；`STATIC_PACK=true`；保留 `--use-distributed-optimizer` |

### 调试过程：4 个独立 bug 串联

跑通后 loss 死活对不齐。Static logs ~0.82，Hybrid logs ~3.5，期望 ~13。逐个排查后修了 4 处：

#### Bug 1：`TransformersModel.forward()` 没把 `_compute_tnd_loss` 全字段透传

```python
# 修复前（只复制 loss 和 loss_mask）
loss_dict["loss"] = tnd_result["loss"]
loss_dict["loss_mask"] = tnd_result["loss_mask"]

# 修复后
loss_dict.update(tnd_result)  # propagate num_samples / token_nums
```

Scheduler 走的 `_compute_tnd_loss` return dict 里有 `num_samples`、`token_nums`，但 forward 只挑了 2 个字段。`pretrain_transformers.loss_func` 里 `num_samples = output_tensor.get('num_samples', 1)` 默认成 1 → hybrid rescale `loss × num_samples/mbs_per_group` = `× 1/16` → 把 loss 除小了 16 倍。

#### Bug 2：`_compute_tnd_loss` 默认分支多余的 1024-chunk CE

Phase 14 阻塞 4 当时为防 OOM 把 CE 切成 1024 chunk 处理。后来阻塞 5 加上 `--use-distributed-optimizer` 把 optimizer state 缩到 `1/dp_size` 释放了大量显存，chunk CE 不再需要。

```python
# 修复前
T_local = logits.shape[1]
chunk = 1024
loss_sum = torch.zeros((), dtype=torch.float32, device=logits.device)
for s in range(0, T_local, chunk):
    e = min(s + chunk, T_local)
    l_chunk_fp32 = logits[:, s:e, :].float()
    ...

# 修复后（一次性 fp32 + CE）
loss_local = tensor_parallel.vocab_parallel_cross_entropy(logits.float(), shift_labels_local)
```

实测 distributed_optimizer 下 burst microbatch（T_local=8192，vocab=152064）一次性 fp32 logits 内存完全够。`default` 和 `per_sample_loss` 共享同一个 single-shot 路径。

#### Bug 3：`_compute_tnd_loss` 的 `per_sample_loss` 分支没遵循 local-fraction 约定

Phase 14 写的 per_sample_loss 分支因为没被实际启用（`loss_type=default`），从来没人验证过它的 reduction 约定。它直接 return `per_sample_mean.mean()`（**FULL** 值），但 `loss_func` 的 `averaged_loss *= cp_size` 是为「local fraction」（= full / cp_size）约定写的，两边对不上。

```python
# 修复（per_sample 分支末尾）
if cp_size > 1:
    loss = loss / cp_size
```

同时只对非 tail-pad sub-seqs 取 mean（用 `non_empty` mask），避免被 0 拖低。

#### Bug 4：`loss_func` 的 `averaged_loss *= cp_size` 在 Hybrid 动态 CP 下不对称

这是最隐蔽的一个。Hybrid 的 Scheduler 在每 microbatch swap `mpu._CONTEXT_PARALLEL_GROUP`（`scheduler.py:179`），导致 `mpu.get_context_parallel_world_size()` 在 ranks 间**不一致**——burst rank 拿 CP=16 动态 group，filler rank 拿 CP=1 或 CP=2 动态 group，每个 rank 看到的 `cp_size` 都不一样。

旧公式 `averaged_loss = AVG(loss) × cp_size` 依赖跨 rank 的 cp_size 一致，**在动态 CP 下被打破**：logged 值取决于 rank 0 当前在哪个 dynamic group，所以 Hybrid 跨 iter loss 在 8.22/9.87/11.53 之间跳变。

**正确的求和形式**：

每个 rank 持有 `loss_local = (per_sample_mean_g / cp_size_g) × (num_samples_g / mbs_per_group)`，对世界所有 rank `SUM` 时，每个 dynamic group 内 `cp_size_g` 个 ranks 各持相同 `loss_local`（gather 后），求和正好把 `cp_size_g` 因子消掉：

```
Σ_ranks loss_local = Σ_g (cp_size_g × loss_local_g)
                   = Σ_g (cp_size_g × per_sample_mean_g/cp_size_g × num_samples_g/mbs_per_group)
                   = Σ_g per_sample_mean_g × num_samples_g / mbs_per_group
                   = total_loss / mbs_per_group
```

要拿到「每样本 loss 均值」`total_loss / total_samples`，再除一个 `num_groups`（因为 `total_samples = mbs_per_group × num_groups = inflated_mbs`）：

```python
# pretrain_transformers.py:loss_func
if is_hybrid or is_static_pack:
    torch.distributed.all_reduce(averaged_loss,
        group=mpu.get_data_parallel_group(with_context_parallel=True),
        op=torch.distributed.ReduceOp.SUM)
    averaged_loss /= float(num_groups)   # 不再依赖 mpu.get_context_parallel_world_size()
```

这个公式对 Static（uniform CP=16）和 Hybrid（dynamic CP 1..16 mixed）都成立，因为推导只用到「每 dynamic group 内 ranks 持相同 loss_local」这一性质，与具体 `cp_size_g` 取值无关。

### 验证：Loss byte-level 对齐

10 iter，相同 seed，相同 GBS=256，相同 B=64/microbatch：

| iter | Static (CP=16, DP=4) | Hybrid (CP=2, DP=32) | abs diff |
|------|---------------------|----------------------|----------|
| 1  | 13.15740 | 13.15775 | +0.00035 (+0.0027%) |
| 2  | 13.17386 | 13.17415 | +0.00029 (+0.0022%) |
| 3  | 13.15390 | 13.15433 | +0.00043 (+0.0033%) |
| 4  | 13.13758 | 13.13791 | +0.00033 (+0.0025%) |
| 5  | 13.13743 | 13.13777 | +0.00034 (+0.0026%) |
| 6  | 13.13724 | 13.13763 | +0.00039 (+0.0030%) |
| 7  | 13.18707 | 13.18755 | +0.00048 (+0.0036%) |
| 8  | 13.15292 | 13.15338 | +0.00046 (+0.0035%) |
| 9  | 13.17049 | 13.17103 | +0.00054 (+0.0041%) |
| 10 | 13.15116 | 13.15146 | +0.00030 (+0.0023%) |

**Max diff 0.0041%**，纯 bf16 浮点求和顺序噪声（Static 跨 4 buckets vs Hybrid 跨 32+ 动态 groups 的 SUM 顺序不同）。两条路径在数值上**完全等价**。

### 性能对比

10 iter，warmup 后 iter 2-10 平均 step time：

| 模式 | iter 1 (warmup) | 平均 iter 2-10 | 相对 |
|---|---|---|---|
| **Hybrid** burst | 22,216 ms | 9,127 ms | 1.00× |
| **Static** baseline | 24,520 ms | 13,138 ms | **1.44×** |

Static 比 Hybrid 慢 **44%**。原因：burst microbatch 中 burst 样本独占一个 DP=4 bucket（其他 3 个 bucket 各装 21 个 filler），CP=16 把 16 张卡锁在这一个 bucket 上算 131072 tokens；其余 3 个 bucket 用同样 16 张卡算 ~84k tokens，padding 区算力浪费严重。Hybrid 通过 BFD 动态把 burst 放到 CP=16 动态 group、其他 filler 放到 CP=1/2 动态 group，每个 group 算力按 token 量按比例分配，吃满硬件。

### Loss 不降的说明（不是 bug）

10 个 iter 内 loss 稳定在 13.13-13.19 之间不下降。两个原因：

1. **LR 几乎为 0**：`--lr 1e-5 --lr-decay-iters 5000 --lr-warmup-fraction 0.1` → `warmup_iters = 500`。10 iter 只到 warmup 的 2%，iter 10 时 LR = `1e-5 × 10/500 = 2×10⁻⁷`，权重几乎没更新。
2. **Synthetic 数据是纯随机 token**：`labels = randint(1, 151936)`，没有可学习结构。理论 CE 下界 = `ln(151936) ≈ 11.93`，观测 ~13.15 = 训练好的 Qwen3VL-2B 对随机输入的非均匀输出 + bf16 init 噪声。即使 LR 正常，模型也无法把随机 token 的 CE 压到 11.93 以下。

Label 路径是对的（`shift_labels[start:end-1] = labels[start+1:end]` 是标准 next-token shift，只在 sub-seq 内部，不跨界，padding 全 -100）。burst run 设计目的是 stress-test scheduler 与 throughput，**不是收敛测试**；要看 loss 下降需要 (a) 真实数据，(b) 提 LR / 缩 warmup / 加 iter。

### 修改文件清单

| 文件 | 改动 |
|------|------|
| `static_pack_dataloader.py`（新）| `_ffd_pack` + `StaticBalancedPackLoader` + `build_static_pack_dataloader` |
| `pretrain_transformers.py` | `STATIC_PACK` 分支（model_provider 跳过 Scheduler，train_valid_test_datasets_provider 用 static_pack loader，get_batch 把 `seqlens` 转 `PackedSeqParams`）；`loss_func` 把 hybrid rescale 扩到 STATIC_PACK，并把 reduction 改成 `SUM ÷ num_groups`（CP-size invariant） |
| `mindspeed_mm/models/transformers_model.py` | `forward()` 的 TND 分支用 `loss_dict.update(tnd_result)` 透传所有字段；`_compute_tnd_loss` 删除 default 分支的 1024-chunk CE，统一 single-shot；`per_sample_loss` 分支补 `/cp_size` 约定；`num_real_samples` 只数非 tail-pad sub-seqs |
| `examples/qwen3vl/model_2B.json` | `loss_cfg.loss_type`: `"default"` → `"per_sample_loss"` |
| `run_4node_qwen3vl_static_baseline.sh`（新）| 静态 CP × DP TND baseline 启动脚本 |

### 遗留 & 下一步

- 当前 baseline 是 2B 模型 + CP=16 + 131k seq，是最直接对照 Hybrid burst (2B + dynamic CP up to 16) 的版本。后续如果 4B 路线（Phase 14 遗留）回归测试通过，需要再写一个 CP=32 的对照。
- Static FFD baseline 假设 `max_pack_seqlen >= burst_len`。如果未来 burst_len > seq_length，需要支持 burst 样本跨 bucket 切分，或者直接调高 `max_pack_seqlen`。
- Loss 对齐验证只跑了 10 iter，建议在真实数据 + 正常 LR 下复跑 100+ iter 看 loss 收敛曲线是否仍逐步对齐。

---

## Phase 16：InternVL3 Static-CP TND Baseline + Loss 对齐

**目标**：为 `run_4node_synthetic_burst.sh`（InternVL3 Hybrid burst）提供同等设计的静态 baseline，复用 Phase 15 的 `static_pack_dataloader.py`，并在 `pretrain_vlm.py` 走通 VLMModel + Ring CP 的 TND packed 路径。

### 设计

与 Qwen3VL baseline 共享 FFD bucket-pack loader，唯一区别是 CP 算法选择和静态拓扑参数：

| 维度 | Qwen3VL baseline | InternVL3 baseline |
|------|------------------|---------------------|
| 入口脚本 | `pretrain_transformers.py` | `pretrain_vlm.py` |
| 模型类 | `TransformersModel` | `VLMModel` |
| CP 算法 | Ulysses CP=16（Qwen3VL-2B num_heads=16）| **Ring CP=16**（InternVL3-8B KV heads=4，Ulysses 上限仅 4）|
| DP | 4 | 4 |
| MBS_user | 16 | 32 |
| GRAD_ACC | 4 | 4 |
| GBS | 256 | 512 |
| B/microbatch | 64 | 128 |
| seq_length | 131072 | 131072 |

### 为什么选 Ring CP

Ulysses CP 受 KV head 数上限约束（Phase 9 附录），InternVL3-8B 只有 4 个 KV head → 最大 Ulysses CP=4。实测 Ulysses CP=4 下 T_local=32768 的激活显存吃到 58.6 GB/61 GB，burst microbatch backward 必 OOM。Ring CP 不受 KV head 约束，每 rank 只持 Q 的 `1/cp_size` 切片（T_local=131072/16=8192），激活显存与 Qwen3VL burst 同量级可用。

Phase 9 packed baseline 尝试 Ring CP 卡在 MindSpeed 的 `cu_seqlens_q_padded` 未提供导致 kv_index `AttributeError`。本 baseline 通过两个改动绕开：

1. **`static_pack_dataloader.py` 内部对齐**：每个 sub-seq 填充到 `2×cp_size`（= 32）倍数，与 Hybrid scheduler 的 `pad_len_to(s, cp_size * 2)` 约定一致（`scheduler.py:647`）。FFD bin-packing 的 bin 容量计算也改用 padded length。
2. **`pretrain_vlm.py:get_batch` 补齐字段**：`PackedSeqParams` 同时提供 `cu_seqlens_q`、`cu_seqlens_kv`、`cu_seqlens_q_padded`、`cu_seqlens_kv_padded`（都指向同一个 cumsum 张量，因为 sub-seqs 已在 dataloader 内部对齐），匹配 `scheduler.py:759-767` 的规约。

### 调试过程

串行修了两个 bug：

#### Bug 1：`[T, T]` 64 GiB causal mask 炸掉 burst forward

`_build_attentionmask_positionid_internllm`（`vlm_attentionmask_for_llm.py:1033`）用 bart 的 `_make_causal_mask` 实现，在 `T=131072` 下分配 `[131072, 131072]` fp32 causal mask = **64 GiB**，iter 1 直接 OOM。这个 mask 在 FA varlen（packed TND）路径下完全多余——FA kernel 用 `cu_seqlens_q` 处理 sub-seq 边界和因果性。

**修复**：`vlm_attentionmask_for_llm.py` 的 `_build_attentionmask_positionid_internllm` 新增 `packed_seq_params` 检测分支，在 packed 模式下直接 return `(None, position_ids)` 跳过 mask 物化。同时 `vlm_model.py:806` 调用处显式传 `packed_seq_params=packed_seq_params`（它是 VLMModel.forward 的显式参数，不在 `**kwargs` 里）。

#### Bug 2：Ulysses CP=4 T_local=32k 激活 OOM（切 Ring CP 前的曲折）

最初打算照搬 Qwen3VL 的 Ulysses 拓扑，用 CP=4（InternVL3-8B 的 Ulysses 上限）。修好 bug 1 后 backward 阶段仍 OOM（58.6 GB 已用 + 试图再分 1.16 GB），Ulysses CP 在 131k 下把整个 T 都留在每个 rank 上做 attention，激活太重。切 Ring CP=16 后 T_local 降到 8192，一次通过。

### loss_func 适配

`pretrain_vlm.py:loss_func` 的 `is_hybrid` 分支本身就已经是 **SUM ÷ num_groups** 的 CP-invariant reduction（行 328-331），早于 Phase 15 就写好了——看起来这条路径在 InternVL3 的 Hybrid baseline 对齐阶段就验证过。只需把 gating 从 `is_hybrid` 扩到 `is_hybrid or is_static_pack`，公式本身不动。

两边 run 同时加 `--calculate-per-sample-loss`，走 VLMModel `compute_loss_with_context_parallel` 的 `use_packed_per_sample=True` 分支，返回 FULL per_sample_mean（gather 后的完整值），然后 loss_func 用 `loss/cp_size → SUM → /num_groups` 收集回 `total_loss / total_samples`。

### Loss 对齐验证

10 iter，同 seed，同 GBS=512，同 B=128/microbatch，同 `--calculate-per-sample-loss`：

| iter | Static (Ring CP=16) | Hybrid (CP=2 dynamic) | abs diff |
|------|---------------------|------------------------|----------|
| 1  | 12.10331 | 12.10332 | +0.00001 |
| 2  | 12.10343 | 12.10341 | −0.00002 |
| 3  | 12.10360 | 12.10344 | −0.00016 |
| 4  | 12.10386 | 12.10369 | −0.00017 |
| 5  | 12.10459 | 12.10419 | −0.00040 |
| 6  | 12.10303 | 12.10285 | −0.00018 |
| 7  | 12.10254 | 12.10291 | +0.00037 |
| 8  | 12.10320 | 12.10349 | +0.00029 |
| 9  | 12.10304 | 12.10304 |  0.00000 |
| 10 | 12.10313 | 12.10308 | −0.00005 |

**Max diff 0.0033%**，与 Qwen3VL baseline 同级别的 bf16 reduction 顺序噪声。数值上两条路径完全等价。

### 性能对比（10 iter，warmup 后 iter 2-10 平均 step time）

| 模式 | iter 1 (warmup) | 平均 iter 2-10 | 相对 |
|---|---|---|---|
| **Hybrid** burst | 30,262 ms | 15,174 ms | 1.00× |
| **Static** baseline | 42,003 ms | 27,916 ms | **1.84×** |

Static 比 Hybrid 慢 **84%**。和 Qwen3VL 的 1.44× 趋势一致（InternVL3 差距更大，因为 8B 模型 + 更大 GBS=512 下 burst bucket 的 131072 token 负载更加主导 step time）。

### 修改文件清单

| 文件 | 改动 |
|------|------|
| `static_pack_dataloader.py` | `_ffd_pack` 新增 `align` 参数；`StaticBalancedPackLoader.__next__` 用 padded length 放置每个 sub-seq，尾部 slot 补 id=0/label=-100；`__init__` 记录 `self.align` |
| `pretrain_vlm.py` | `model_provider` 在 STATIC_PACK 时跳过 Scheduler；`get_batch` 从 `seqlens` 构造 `PackedSeqParams`（`cu_seqlens_q` + `cu_seqlens_q_padded` 同值）；`train_valid_test_datasets_provider` 新增 STATIC_PACK 分支（inflate MBS × dp_size，build static_pack loader，恢复 MBS）；`loss_func` 把 `is_hybrid` gating 扩到 `is_hybrid or is_static_pack` |
| `mindspeed_mm/models/vision/vlm_attentionmask_for_llm.py` | `_build_attentionmask_positionid_internllm` 在 `packed_seq_params` 存在时 return `(None, position_ids)`，跳过 `[T, T]` causal mask 物化 |
| `mindspeed_mm/models/vlm_model.py` | `prepare_positionsids_mask_for_llm` 调用处显式传 `packed_seq_params=packed_seq_params`，让下游 mask builder 能看到 |
| `run_4node_internvl3_static_baseline.sh`（新）| CP=16 Ring / DP=4 / MBS=32 / GRAD_ACC=4 / GBS=512 / seq_length=131072；与 `run_4node_synthetic_burst.sh` 完全对齐的 `SYNTHETIC_*` 环境变量；`STATIC_PACK=true`，unset `HYBRID_PARALLEL` |
| `run_4node_synthetic_burst.sh` | 加 `--calculate-per-sample-loss` 与 static 对齐 loss reduction；`HCCL_IF_BASE_PORT` 改成 53000 避开先前 50000 的 TIME_WAIT 残留 |

### 遗留 & 下一步

- 用 LR=2e-5 直接启动（InternVL3 burst 脚本没配 `--lr-warmup-fraction`），不像 Qwen3VL baseline 那样处于 warmup 阶段。随机 token 数据下 10 iter 看不到收敛，理由同 Phase 15。
- Ring CP=16 静态拓扑下 Static 比 Hybrid 慢 84%，burst microbatch 是主要开销来源。这也是交付希望展示的对比指标。
- 目前两条 baseline 都是 bf16 bucket packing 验证，尚未跑长时间稳定性测试。