#!/bin/bash
# launch_4node.sh
# 仅在主节点运行，自动在所有节点启动训练
#
# 用法:
#   bash launch_4node.sh                          # 使用默认训练脚本
#   bash launch_4node.sh finetune_xxx.sh          # 指定训练脚本
#
# 日志:
#   主节点: 训练脚本内自行写 logs/
#   Worker: /tmp/train_worker_<IP>_<timestamp>.log

WORKDIR="/data/user/user40/hybrid_parallel/MindSpeed-MM"
TRAIN_SCRIPT="${1:-finetune_internvl3_8B_hybrid_multi_node.sh}"
HOSTFILE="$WORKDIR/examples/internvl3/hostfile.txt"
CONDA_PROFILE="/data/user/user40/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/opt/conda/private/envs/MindSpeed-MM"
LAUNCH_SCRIPT_DIR="$WORKDIR/.launcher"

# ── 校验：必须在主节点运行 ─────────────────────────────────────────────────────
MASTER_IP=$(head -n1 "$HOSTFILE")
LOCAL_IP=""
for ip in $(hostname -I); do
    if [ "$ip" = "$MASTER_IP" ]; then
        LOCAL_IP=$ip
        break
    fi
done

if [ -z "$LOCAL_IP" ]; then
    echo "[launcher] 错误：本脚本必须在主节点（IP: $MASTER_IP）上运行"
    echo "[launcher] 本机 IP 列表: $(hostname -I)"
    exit 1
fi

NNODES=$(wc -l < "$HOSTFILE")
LAUNCH_TS=$(date +%Y%m%d_%H%M%S)

echo "[launcher] ============================================"
echo "[launcher] 主节点: $MASTER_IP"
echo "[launcher] 节点总数: $NNODES"
echo "[launcher] 训练脚本: $TRAIN_SCRIPT"
echo "[launcher] ============================================"

# ── 收集 worker IP（hostfile 去掉第一行）─────────────────────────────────────
WORKER_IPS=()
while IFS= read -r ip; do
    [ -n "$ip" ] && [ "$ip" != "$MASTER_IP" ] && WORKER_IPS+=("$ip")
done < <(tail -n +2 "$HOSTFILE")

# ── Ctrl-C 时清理所有节点的训练进程 ─────────────────────────────────────────
cleanup() {
    echo ""
    echo "[launcher] 收到中断，正在停止所有 worker..."
    pkill -f pretrain_vlm.py 2>/dev/null
    pkill -f pretrain_transformers.py 2>/dev/null
    for ip in "${WORKER_IPS[@]}"; do
        ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@"$ip" \
            "tmux kill-session -t train 2>/dev/null; pkill -f pretrain_vlm.py 2>/dev/null; pkill -f pretrain_transformers.py 2>/dev/null" &
    done
    wait
    exit 130
}
trap cleanup QUIT TERM

# ── 为每个 worker 在共享文件系统上生成独立启动脚本（避开 SSH 引号嵌套问题）──
mkdir -p "$LAUNCH_SCRIPT_DIR"

# Forward selected env vars to workers so per-run overrides
# (SYNTHETIC_BURST_LEN, SCHED_SEQ_LEN_CHUNK, STATIC_SEQ_LEN, TRAIN_ITERS, RUN_TAG, ...)
# propagate into the worker shell.
FORWARD_ENV=""
for var in SYNTHETIC_BURST_LEN SYNTHETIC_TOKEN_BUDGET_PER_GPU SCHED_SEQ_LEN_CHUNK \
           STATIC_SEQ_LEN TRAIN_ITERS RUN_TAG SCHEDULE_MODE HYBRID_PARALLEL \
           SYNTHETIC_LENGTH_DIST SYNTHETIC_BURST_PROB SYNTHETIC_NUM_BATCHES \
           MASTER_PORT HCCL_IF_BASE_PORT LR CP MBS GRAD_ACC_STEP; do
    val="${!var}"
    if [ -n "$val" ]; then
        FORWARD_ENV+="export ${var}=${val}"$'\n'
    fi
done

for ip in "${WORKER_IPS[@]}"; do
    REMOTE_LOG="/tmp/train_worker_${ip}_${LAUNCH_TS}.log"
    WORKER_SCRIPT="$LAUNCH_SCRIPT_DIR/worker_${ip}.sh"

    cat > "$WORKER_SCRIPT" << WORKEREOF
#!/bin/bash
# 直接 prepend conda env bin 到 PATH，不依赖 conda activate（private env 在 worker 上未注册）
export PATH=$CONDA_ENV/bin:\$PATH
export LD_LIBRARY_PATH=$CONDA_ENV/lib:\$LD_LIBRARY_PATH
${FORWARD_ENV}cd $WORKDIR
bash $TRAIN_SCRIPT 2>&1 | tee $REMOTE_LOG
WORKEREOF
    chmod +x "$WORKER_SCRIPT"

    echo "[launcher] 启动 worker $ip  (log -> root@${ip}:${REMOTE_LOG})"
    ssh -n -o StrictHostKeyChecking=no root@"$ip" \
        "tmux kill-session -t train 2>/dev/null; tmux new-session -d -s train 'bash $WORKER_SCRIPT'"
done

echo "[launcher] 所有 worker 已在后台启动，2 秒后启动主节点..."
sleep 2

# ── 主节点前台运行 ────────────────────────────────────────────────────────────
cd "$WORKDIR"
source "$CONDA_PROFILE"
conda activate "$CONDA_ENV"
bash "$TRAIN_SCRIPT"
