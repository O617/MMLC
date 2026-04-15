#!/bin/bash
# run_4node_internvl3_static_baseline.sh
# InternVL3 4 节点 64 卡 静态 CP × DP TND Packing baseline
# 对照 run_4node_synthetic_burst.sh（动态 Hybrid 调度）
#
# 设计（mirror Phase 15 Qwen3VL static baseline）：
#   - 不走 Scheduler / HybridScheduledDataLoader
#   - CP 在启动时静态设定，覆盖 burst 最大序列 131072
#   - 与 Hybrid burst 完全一致的样本/数据口径：
#       Hybrid burst: MBS_user=4, GRAD_ACC=4, num_groups=DP_dyn=32, GBS = 4×4×32 = 512
#                     每 microbatch 从 SyntheticDataLoader 取 B = 4×32 = 128 samples
#       Static base : MBS_user=32, GRAD_ACC=4, DP_static=4,         GBS = 32×4×4 = 512 ✓
#                     每 microbatch 从 SyntheticDataLoader 取 B = 32×4 = 128 samples ✓
#     两条路径在同 seed 下 loader.__next__ 产出 byte-identical 的 128 个样本，
#     total samples per step 也都是 512，唯一差别是样本到 rank 的分配方式。
#
# CP=16 Ring（megatron_cp_algo）：
#   - Ulysses CP 受 InternVL3-8B 的 4 个 KV head 限制，上限=4；但 CP=4 Ulysses
#     下 T_local=32k 激活太大（实测 ~59 GB），吃不下 burst microbatch。
#   - Ring CP 不受 kv_head 约束，每 rank 只持有 Q 的 1/cp_size 切片，激活显存
#     大幅下降（T_local=131072/16=8192 per rank，与 Qwen3VL burst 同量级）。
#   - Phase 9 的 Ring + packing 阻塞点是 MindSpeed 的 `cu_seqlens_q_padded`
#     未提供导致 kv_index 未计算抛 AttributeError。本 baseline 把每个 sub-seq
#     在 static_pack_dataloader 内部 pad 到 2×CP=32 倍数，同时在 get_batch 里
#     同时提供 `cu_seqlens_q` 和 `cu_seqlens_q_padded`，匹配 Hybrid scheduler
#     的 PackedSeqParams 规约（scheduler.py:759-767），走通 Ring 的 packed 路径。
#
# seq_length = 131072 = burst_len 是 packing cap：
#   FFD bin-pack 把 128 个样本装进 16 个 bucket（每 bucket cap=131072）：
#     burst microbatch  → 1 bucket = [burst(131072)] 独占；
#                         其余 15 bucket 各 ~127k tokens / ~8 fillers
#     非 burst microbatch → 16 bucket × 8 samples ≈ ~25-32k 各
#   每 bucket 输出固定形状 [1, 131072]，尾部全 padding(label=-100)。
#   131072 已被 8 (= 2×CP) 整除 → Ulysses zigzag 对齐天然成立。

export LD_LIBRARY_PATH=/opt/conda/private/envs/MindSpeed-MM/lib:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=600
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000
export ASCEND_LAUNCH_BLOCKING=1
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-53000}
export TOKENIZERS_PARALLELISM=false

# ── 静态 CP × DP baseline 开关 ────────────────────────────────────────────────
unset HYBRID_PARALLEL
unset SCHEDULE_MODE
export STATIC_PACK=true

# ── 合成数据配置（与 Hybrid burst 完全对齐）──────────────────────────────────
export SYNTHETIC_DATA=True
export SYNTHETIC_LENGTH_DIST=${SYNTHETIC_LENGTH_DIST:-burst}

export SYNTHETIC_BURST_PROB=0.5
# 128k variant: 131072; 256k variant: 229376 (≈224k, within 200-240k acceptable band)
export SYNTHETIC_BURST_LEN=${SYNTHETIC_BURST_LEN:-131072}
export SYNTHETIC_CLUSTER_SIZE=64
export SYNTHETIC_TOKEN_BUDGET_PER_GPU=${SYNTHETIC_TOKEN_BUDGET_PER_GPU:-8192}
export SYNTHETIC_BURST_NOISE=0.1
export SYNTHETIC_BURST_FALLBACK=uniform

export SYNTHETIC_MIN_LEN=512
export SYNTHETIC_MAX_LEN=4096
export SYNTHETIC_NUM_BATCHES=200

# Vision 配置（与 Hybrid burst 一致；ratio=0.01 → 普通序列 0 tiles，burst 5 tiles）
export SYNTHETIC_WITH_VISION=true
export SYNTHETIC_VISION_RATIO=0.01
export SYNTHETIC_TILE_TOKENS=256
export SYNTHETIC_TILE_HW=448
export SYNTHETIC_IMG_TOKEN_ID=151667

# ── 集群拓扑 ──────────────────────────────────────────────────────────────────
NPUS_PER_NODE=16
HOSTFILE="examples/internvl3/hostfile.txt"
MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1}')
MASTER_PORT=6000

LOCAL_IPS=$(hostname -I)
NODE_ADDR=""
NODE_RANK=""
for ip in $LOCAL_IPS; do
    rank=$(awk -v ip="$ip" '$1 == ip {print NR-1; exit}' $HOSTFILE)
    if [ -n "$rank" ]; then
        NODE_ADDR=$ip
        NODE_RANK=$rank
        break
    fi
done

if [ -z "$NODE_ADDR" ]; then
    echo "错误：无法在 $HOSTFILE 中找到本机 IP"
    exit 1
fi

NNODES=$(wc -l < $HOSTFILE)
WORLD_SIZE=$(( NPUS_PER_NODE * NNODES ))

TP=1
PP=1
CP=${CP:-16}
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
# samples/microbatch on each DP rank stays 128 (= hybrid MBS_user×num_groups)
# → MBS = 128 / DP. CP=16→DP=4→MBS=32; CP=32→DP=2→MBS=64
MBS=${MBS:-$((128/$DP))}
GRAD_ACC_STEP=${GRAD_ACC_STEP:-4}
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

echo "[internvl3-static] WORLD_SIZE=$WORLD_SIZE  CP=$CP  DP=$DP  MBS=$MBS  GRAD_ACC=$GRAD_ACC_STEP  GBS=$GBS"
echo "[internvl3-static] BURST_LEN=$SYNTHETIC_BURST_LEN  BURST_PROB=$SYNTHETIC_BURST_PROB"
echo "[internvl3-static] seq_length=131072  inflated_B/microbatch=$(($MBS*$DP))"

MM_MODEL="./examples/internvl3/model_8B.json"
MM_DATA="./examples/internvl3/data_8B_hybrid.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="./ckpt/mm_path/internvl3-8B"
SAVE_PATH="save_dir_internvl3_static_baseline"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
    --use-cp-send-recv-overlap \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --seq-length ${STATIC_SEQ_LEN:-131072} \
    --max-position-embeddings ${STATIC_SEQ_LEN:-131072} \
    --tokenizer-type NullTokenizer \
    --vocab-size 151674 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 2e-6 \
    --min-lr 0.0 \
    --train-iters ${TRAIN_ITERS:-30} \
    --lr-decay-iters 5000 \
    --lr-decay-style cosine \
    --weight-decay 0.05 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --use-distributed-optimizer \
    --use-flash-attn \
    --bf16 \
    --load $LOAD_PATH \
    --variable-seq-lengths \
    --normalization RMSNorm \
    --num-workers 0 \
    --calculate-per-sample-loss \
"

MM_ARGS="
    --mm-data ${MM_DATA} \
    --mm-model ${MM_MODEL} \
    --mm-tool ${MM_TOOL}
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --eval-iters 5000 \
    --save $SAVE_PATH \
    --ckpt-format torch \
    --log-tps \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS \
    pretrain_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/${RUN_TAG:-train_internvl3_static_baseline}_${logfile}.log 2>&1
