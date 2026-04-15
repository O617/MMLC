#!/bin/bash
# run_4node_synthetic_burst.sh
# 4 节点 64 卡 Hybrid + Burst 合成数据测试
# 用法: bash launch_4node.sh run_4node_synthetic_burst.sh

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
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-53000}

# ── Hybrid Parallel 开关 ──────────────────────────────────────────────────────
export HYBRID_PARALLEL=True
export SCHEDULE_MODE=dynamic
export USE_HYBRID_DATALOADER=0

# ── 合成数据配置 ──────────────────────────────────────────────────────────────
export SYNTHETIC_DATA=True
export SYNTHETIC_LENGTH_DIST=burst

# Burst 参数
export SYNTHETIC_BURST_PROB=0.5           # 50% 概率触发，方便在少量 step 内观察到 burst
export SYNTHETIC_BURST_LEN=${SYNTHETIC_BURST_LEN:-131072}         # 128k / 256k variants
export SYNTHETIC_CLUSTER_SIZE=64
export SYNTHETIC_TOKEN_BUDGET_PER_GPU=${SYNTHETIC_TOKEN_BUDGET_PER_GPU:-8192}
export SYNTHETIC_BURST_NOISE=0.1
export SYNTHETIC_BURST_FALLBACK=uniform

# 非 burst 步的序列长度范围（保证总长不超预算）
# batch_size = MBS × num_groups = 4 × 32 = 128
# max_total = 128 × 4096 = 524,288 = budget
export SYNTHETIC_MIN_LEN=512
export SYNTHETIC_MAX_LEN=4096
export SYNTHETIC_NUM_BATCHES=200

# Vision 配置（必须开启以避免 Scheduler image_flags 访问 KeyError）
# ratio=0.01: 普通序列(~3k token) -> 0 tiles；burst 序列(128k) -> 5 tiles (~12MB)
export SYNTHETIC_WITH_VISION=true
export SYNTHETIC_VISION_RATIO=0.01
export SYNTHETIC_TILE_TOKENS=256
export SYNTHETIC_TILE_HW=448
export SYNTHETIC_IMG_TOKEN_ID=151667

# ── 集群拓扑 ──────────────────────────────────────────────────────────────────
NPUS_PER_NODE=16
HOSTFILE="examples/internvl3/hostfile.txt"
MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1}')
MASTER_PORT=${MASTER_PORT:-6000}

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

MBS=4
GRAD_ACC_STEP=4
TP=1
PP=1
CP=2
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

echo "[synthetic-burst] WORLD_SIZE=$WORLD_SIZE  DP=$DP  GBS=$GBS"
echo "[synthetic-burst] BURST_LEN=$SYNTHETIC_BURST_LEN  BURST_PROB=$SYNTHETIC_BURST_PROB"
echo "[synthetic-burst] token_budget=$(($SYNTHETIC_CLUSTER_SIZE * $SYNTHETIC_TOKEN_BUDGET_PER_GPU))"

MM_MODEL="./examples/internvl3/model_8B.json"
MM_DATA="./examples/internvl3/data_8B_hybrid.json"   # 合成模式下不实际读取
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="./ckpt/mm_path/internvl3-8B"
SAVE_PATH="save_dir_synthetic"

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
    --use-cp-send-recv-overlap \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --seq-length 4096 \
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
    | tee logs/${RUN_TAG:-train_synthetic_burst}_${logfile}.log 2>&1
