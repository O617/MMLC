#!/bin/bash
# run_4node_qwen3vl_synthetic_burst.sh
# Qwen3VL 4 节点 64 卡 Hybrid + Burst 合成数据测试
# 参数设定与 run_4node_synthetic_burst.sh（InternVL3）对齐
# 用法: bash launch_4node.sh run_4node_qwen3vl_synthetic_burst.sh

export LD_LIBRARY_PATH=/opt/conda/private/envs/MindSpeed-MM/lib:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000
export ASCEND_LAUNCH_BLOCKING=1
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-52000}
export TOKENIZERS_PARALLELISM=false

# ── Hybrid Parallel 开关 ──────────────────────────────────────────────────────
export HYBRID_PARALLEL=True
export SCHEDULE_MODE=${SCHEDULE_MODE:-dynamic}
export USE_HYBRID_DATALOADER=0

# ── 合成数据配置（与 InternVL3 burst 测试参数一致）──────────────────────────────
export SYNTHETIC_DATA=True
export SYNTHETIC_LENGTH_DIST=${SYNTHETIC_LENGTH_DIST:-burst}

export SYNTHETIC_BURST_PROB=0.5
export SYNTHETIC_BURST_LEN=${SYNTHETIC_BURST_LEN:-131072}
export SYNTHETIC_CLUSTER_SIZE=64
# For 128k burst keep chunk=8192; for 256k raise to >=16384 (+margin)
export SYNTHETIC_TOKEN_BUDGET_PER_GPU=${SYNTHETIC_TOKEN_BUDGET_PER_GPU:-8192}
export SCHED_SEQ_LEN_CHUNK=${SCHED_SEQ_LEN_CHUNK:-8192}
export SYNTHETIC_BURST_NOISE=0.1
export SYNTHETIC_BURST_FALLBACK=uniform

# B = MBS(2) × num_groups(32) = 64 samples/microbatch
# cluster_max = SYNTHETIC_CLUSTER_SIZE × SYNTHETIC_TOKEN_BUDGET_PER_GPU = 524288
# burst(131072) + 63 filler × ~4096 ≤ 524288 ✓
export SYNTHETIC_MIN_LEN=512
export SYNTHETIC_MAX_LEN=4096
export SYNTHETIC_NUM_BATCHES=200

# 纯文本合成：Phase 13 已在 scheduler 加入空 image_flags 的 fast path，
# 无需开启 vision（Qwen3VL 需要 image_grid_thw，而 synthetic_dataloader 不产出该字段）。
export SYNTHETIC_WITH_VISION=false

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

# Qwen3VL 调度可行性约束：
#   scheduler.seq_len_chunk = 8192（see pretrain_transformers.model_provider）
#   cluster_max_tokens = 64 × 8192 = 524288 tokens/microbatch
#   B = MBS × num_groups = 2 × 32 = 64 samples/microbatch
#   burst(256k) 吃满 CP=32；63 个 filler 按 ~4k/个 → 32 个 CP=1 bin
#   总 rank 需求 = 32 + 32 = 64 ✓
# MBS=4 时 filler 数翻倍（127→64 bin），rank 预算爆掉，burst 被迫降 CP，无法测 128k/256k
MBS=2
GRAD_ACC_STEP=4
TP=1
PP=1
CP=2
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

echo "[qwen3vl-burst] WORLD_SIZE=$WORLD_SIZE  DP=$DP  GBS=$GBS"
echo "[qwen3vl-burst] BURST_LEN=$SYNTHETIC_BURST_LEN  BURST_PROB=$SYNTHETIC_BURST_PROB"
echo "[qwen3vl-burst] token_budget=$(($SYNTHETIC_CLUSTER_SIZE * $SYNTHETIC_TOKEN_BUDGET_PER_GPU))"

MM_MODEL="./examples/qwen3vl/model_2B.json"
MM_DATA="./examples/qwen3vl/data_8B_hybrid.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="./ckpt/hf_path/Qwen3VL-2B-Instruct"
SAVE_PATH="save_dir_qwen3vl_synthetic"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --seq-length 4096 \
    --tokenizer-type NullTokenizer \
    --vocab-size 152064 \
    --make-vocab-size-divisible-by 1 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --lr ${LR:-1.0e-6} \
    --min-lr 0.0 \
    --train-iters ${TRAIN_ITERS:-30} \
    --lr-decay-iters 5000 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.0 \
    --weight-decay 0 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --seed 42 \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --use-flash-attn \
    --use-distributed-optimizer \
    --bf16 \
    --load $LOAD_PATH \
    --variable-seq-lengths \
    --untie-embeddings-and-output-weights \
    --optimizer-selection fused_torch_adamw \
    --use-cpu-initialization \
    --num-workers 0 \
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
    pretrain_transformers.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/${RUN_TAG:-train_qwen3vl_synthetic_burst}_${logfile}.log 2>&1
