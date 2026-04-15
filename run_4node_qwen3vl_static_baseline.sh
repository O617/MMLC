#!/bin/bash
# run_4node_qwen3vl_static_baseline.sh
# Qwen3VL 4 节点 64 卡 静态 CP × DP TND Packing baseline
# 对照 run_4node_qwen3vl_synthetic_burst.sh（动态 Hybrid 调度）
#
# 设计：
#   - 不走 Scheduler / HybridScheduledDataLoader
#   - CP 在启动时静态设定为 burst 最大序列所需值（128k → CP=16 on 2B）
#   - DP = WORLD_SIZE / CP = 64 / 16 = 4
#   - 每个 microbatch 为一个 [1, seq_length] TND packed 张量
#   - static_pack_dataloader 在线 bin-pack 短序列至 seq_length，多余补 pad
#
# Token budget 对齐：
#   per-microbatch tokens = seq_length = 131072
#   per-step tokens       = MBS(1) × GRAD_ACC(4) × DP(4) × 131072 = 2,097,152
#   （Hybrid burst 单 step token budget ~524k；本静态 baseline 因每 microbatch 强行
#   填满 128k 而比 Hybrid 多算 ~4×，这正是静态 CP 在短序列工作负载下的代价，
#   也是这次对照要展示的核心）

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
export HCCL_CONNECT_TIMEOUT=600
export TOKENIZERS_PARALLELISM=false

# ── 静态 CP × DP baseline 开关 ────────────────────────────────────────────────
# HYBRID_PARALLEL 显式不导出（pretrain_transformers.py 走默认 False 路径）
unset HYBRID_PARALLEL
unset SCHEDULE_MODE
export STATIC_PACK=true

# ── 合成数据配置（与 burst 测试参数完全对齐，唯一不同是 LOADER 行为）──────────
export SYNTHETIC_LENGTH_DIST=${SYNTHETIC_LENGTH_DIST:-burst}

export SYNTHETIC_BURST_PROB=0.5
# 128k variant: 131072; 256k variant: 229376 (≈224k, fits in 200-240k acceptable band)
export SYNTHETIC_BURST_LEN=${SYNTHETIC_BURST_LEN:-131072}
export SYNTHETIC_CLUSTER_SIZE=64
export SYNTHETIC_TOKEN_BUDGET_PER_GPU=${SYNTHETIC_TOKEN_BUDGET_PER_GPU:-8192}
export SYNTHETIC_BURST_NOISE=0.1
export SYNTHETIC_BURST_FALLBACK=uniform

export SYNTHETIC_MIN_LEN=512
export SYNTHETIC_MAX_LEN=4096
export SYNTHETIC_NUM_BATCHES=200

# 纯文本（与 Hybrid burst 一致）
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

# 静态 CP × DP baseline，与 Hybrid burst 完全对齐的样本/数据口径：
#   Hybrid burst: MBS_user=2, GRAD_ACC=4, num_groups=DP_dyn=32, GBS = 2×4×32 = 256
#                 每 microbatch 从 SyntheticDataLoader 取 B = 2×32 = 64 samples
#   Static base : MBS_user=16, GRAD_ACC=4, DP_static=4,   GBS = 16×4×4 = 256 ✓
#                 每 microbatch 从 SyntheticDataLoader 取 B = 16×4 = 64 samples ✓
#   两条路径在同 seed 下 loader.__next__ 产出 byte-identical 的 64 个样本，
#   total samples per step 也都是 256，唯一差别是样本到 rank 的分配方式。
#
# CP=16 → Ulysses 内每节点闭环（16 卡刚好是 Qwen3VL-2B 的 num_attention_heads
# 上限），跨节点只走 DP allreduce。
#
# seq_length = 131072 = burst_len 是 packing cap：
#   FFD bin-pack 把 64 个样本装进 4 个 bucket（每 bucket cap=131072）：
#     burst microbatch  → bucket0 = [burst(131072)]  独占；
#                         bucket1-3 ≈ 21 fillers ≈ 84k 各
#     非 burst microbatch → 4 bucket × ~16 samples ≈ ~40-50k 各
#   每 bucket 输出固定形状 [1, 131072]，尾部全 padding(label=-100)。
#   131072 已被 32 (= 2×CP) 整除 → Ulysses zigzag 对齐天然成立。
TP=1
PP=1
CP=16
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
MBS=16
GRAD_ACC_STEP=4
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

echo "[qwen3vl-static] WORLD_SIZE=$WORLD_SIZE  CP=$CP  DP=$DP  MBS=$MBS  GRAD_ACC=$GRAD_ACC_STEP  GBS=$GBS"
echo "[qwen3vl-static] BURST_LEN=$SYNTHETIC_BURST_LEN  BURST_PROB=$SYNTHETIC_BURST_PROB"
echo "[qwen3vl-static] seq_length=131072  inflated_B/microbatch=$(($MBS*$DP))"

MM_MODEL="./examples/qwen3vl/model_2B.json"
MM_DATA="./examples/qwen3vl/data_8B_hybrid.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="./ckpt/hf_path/Qwen3VL-2B-Instruct"
SAVE_PATH="save_dir_qwen3vl_static_baseline"

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
    --seq-length ${STATIC_SEQ_LEN:-131072} \
    --max-position-embeddings ${STATIC_SEQ_LEN:-131072} \
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
    | tee logs/${RUN_TAG:-train_qwen3vl_static_baseline}_${logfile}.log 2>&1
