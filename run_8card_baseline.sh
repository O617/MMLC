#!/bin/bash
export LD_LIBRARY_PATH=/opt/conda/private/envs/MindSpeed-MM/lib:
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/user/user40/miniconda3/bin/activate
conda activate /opt/conda/private/envs/MindSpeed-MM

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
export HCCL_BUFFSIZE=200

NPUS_PER_NODE=8
NNODES=1
WORLD_SIZE=8

MBS=2
GRAD_ACC_STEP=8
TP=1
PP=1
CP=2
DP=$(($WORLD_SIZE/$TP/$PP/$CP))    # DP=4
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))   # GBS=64

MM_DATA='./examples/internvl3/data_8B.json'
MM_MODEL='./examples/internvl3/model_8B.json'
MM_TOOL='./mindspeed_mm/tools/tools.json'
LOAD_PATH='./ckpt/mm_path/internvl3-8B'
SAVE_PATH='save_dir'

mkdir -p logs
logfile=baseline_8card_$(date +%Y%m%d_%H%M%S)

torchrun --nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --master_port 30000     pretrain_vlm.py     --tensor-model-parallel-size ${TP}     --pipeline-model-parallel-size ${PP}     --context-parallel-size ${CP}     --micro-batch-size ${MBS}     --global-batch-size ${GBS}     --seq-length 4096     --tokenizer-type NullTokenizer     --vocab-size 151674     --position-embedding-type rope     --rotary-base 1000000     --swiglu     --no-masked-softmax-fusion     --lr 2e-5     --min-lr 0.0     --train-iters 20     --lr-decay-style cosine     --weight-decay 0.05     --clip-grad 1.0     --adam-beta1 0.9     --adam-beta2 0.999     --no-gradient-accumulation-fusion     --no-load-optim     --no-load-rng     --no-save-optim     --no-save-rng     --use-distributed-optimizer     --use-flash-attn     --bf16     --variable-seq-lengths     --normalization RMSNorm     --num-workers 2     --calculate-per-sample-loss     --log-interval 1     --save-interval 5000     --eval-interval 5000     --eval-iters 5000     --save $SAVE_PATH     --ckpt-format torch     --mm-data ${MM_DATA}     --mm-model ${MM_MODEL}     --mm-tool ${MM_TOOL}     --distributed-backend nccl     2>&1 | tee logs/${logfile}.log

echo Log saved to logs/${logfile}.log
