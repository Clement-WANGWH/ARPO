#!/bin/bash

#================== Basic Configuration ==================#
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 获取脚本所在的目录，并设置正确的PYTHONPATH，使其包含 src 目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$(dirname "$SCRIPT_DIR")/src":$PYTHONPATH

# 禁用 Weights & Biases
export WANDB_DISABLED=true

#================== Training Parameter Configuration ==================#
NNODES=1                 # 节点总数
NODE_RANK=0              # 当前节点的排名
PROC_PER_NODE=6          # 每个节点的进程数 (应与 CUDA_VISIBLE_DEVICES 中的 GPU 数量一致)
MASTER_ADDR="127.0.0.1"  # 主节点地址
MASTER_PORT=29500        # 主节点端口

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/checkpoint/Qwen2.5-7B-Instruct"
mkdir -p ${OUTPUT_DIR}

# 训练脚本的路径
TRAIN_SCRIPT="../src/llamafactory/launcher.py"

# 训练参数配置文件的路径
TRAIN_ARGS="yaml/qwen.yaml"

echo "Starting training..."
torchrun --nnodes ${NNODES} \
         --node_rank ${NODE_RANK} \
         --nproc_per_node ${PROC_PER_NODE} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         ${TRAIN_SCRIPT} ${TRAIN_ARGS} 2>&1 | tee ${OUTPUT_DIR}/training.log
