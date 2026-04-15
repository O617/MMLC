#!/bin/bash
# run_4node_packed_baseline.sh
# 4 节点 64 卡 静态 CP + 原生 Packing baseline
#
# 对标对象：run_4node_synthetic_burst.sh（Dynamic Hybrid, CP=2 dynamic）
#
# 设计原则（Token-based GBS）：
#   - CP=32（Ring Attention, megatron_cp_algo），DP=2
#   - DynamicBatching 将变长序列 bin-pack 到 max_seq_len≈128k tokens/microbatch
#   - MBS=1（1 个 packed 序列/microbatch）, GRAD_ACC=2
#   - 每次 optimizer step 消耗 ≈ 128k × 2 × 2 = 512k tokens（与 burst test 总 budget 对齐）
#   - GBS = MBS × GRAD_ACC × DP = 1 × 2 × 2 = 4
#
# 用法: bash launch_4node.sh run_4node_packed_baseline.sh

# ── 环境变量 ──────────────────────────────────────────────────────────────────
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
export HCCL_IF_BASE_PORT=50000

# ── Baseline 标志：关闭 Hybrid Parallel ──────────────────────────────────────
# HYBRID_PARALLEL 不设置 → 走标准静态 CP 路径

# ── 合成数据（与 burst test 保持一致）────────────────────────────────────────
export SYNTHETIC_DATA=True
export SYNTHETIC_LENGTH_DIST=burst

export SYNTHETIC_BURST_PROB=0.5
export SYNTHETIC_BURST_LEN=131072          # 128k tokens
export SYNTHETIC_CLUSTER_SIZE=64
export SYNTHETIC_TOKEN_BUDGET_PER_GPU=8192
export SYNTHETIC_BURST_NOISE=0.1
export SYNTHETIC_BURST_FALLBACK=uniform

export SYNTHETIC_MIN_LEN=512
export SYNTHETIC_MAX_LEN=4096
export SYNTHETIC_NUM_BATCHES=2000          # 足够覆盖多个 epoch
export SYNTHETIC_WITH_VISION=false         # 文本 only，避免 image_grid_thw 依赖

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

# ── 并行度（Token-based GBS）─────────────────────────────────────────────────
TP=1
PP=1
CP=32                                      # 静态 Ring Attention CP，最大支持 128k 序列
DP=$(( WORLD_SIZE / TP / PP / CP ))        # = 64 / 32 = 2
MAX_SEQ_LEN=131072                         # bin-pack 目标长度（128k tokens/microbatch）
MBS=1                                      # 1 个 packed 序列/microbatch
GRAD_ACC=2                                 # 每个 optimizer step 2 个 microbatch/DP 副本
GBS=$(( MBS * GRAD_ACC * DP ))             # = 1 × 2 × 2 = 4

echo "[packed-baseline] WORLD_SIZE=$WORLD_SIZE  CP=$CP  DP=$DP"
echo "[packed-baseline] MAX_SEQ_LEN=$MAX_SEQ_LEN  MBS=$MBS  GRAD_ACC=$GRAD_ACC  GBS=$GBS"
echo "[packed-baseline] token_budget/step ≈ $((MAX_SEQ_LEN * GRAD_ACC * DP)) tokens"

# ── 路径 ──────────────────────────────────────────────────────────────────────
MM_MODEL="./examples/internvl3/model_8B.json"
MM_DATA="./examples/internvl3/data_8B_hybrid.json"   # 合成模式下不实际读取
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="./ckpt/mm_path/internvl3-8B"
SAVE_PATH="save_dir_packed_baseline"

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
    --seq-length ${MAX_SEQ_LEN} \
    --tokenizer-type NullTokenizer \
    --vocab-size 151674 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 2e-5 \
    --min-lr 0.0 \
    --train-iters 10 \
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
    --use-txt-dynamic-batching \
    --max-seq-len ${MAX_SEQ_LEN} \
    --dynamic-batch-buffer-size 200 \
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
    | tee logs/train_packed_baseline_${logfile}.log 2>&1

