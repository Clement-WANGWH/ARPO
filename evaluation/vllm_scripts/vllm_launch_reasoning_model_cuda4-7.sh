#!/bin/bash

use_qwen3=false

# Activate the Conda environment
source /root/miniconda3/etc/profile.d/conda.sh
#source /root/miniconda3/envs/vllm_env/bin/activate
conda activate sft

# Move to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "cd $SCRIPT_DIR"

# Create log directory
mkdir -p logs

# Model path - same model used for all instances
MODEL_PATH="/root/autodl-tmp/Qwen2.5-7B-Instruct-sft"
MODEL_NAME="Qwen2.5-7B-Instruct-sft"

# Launch instance 1 - using GPU 0
echo "Starting Instance 1 on GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.9 \
    --port 8002 > logs/model0.log 2>&1 &
INSTANCE1_PID=$!
echo "Instance 1 deployed on port 8002 using GPU 0"

# Launch instance 2 - using GPU 1
CUDA_VISIBLE_DEVICES=1 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.9 \
    --port 8003 > logs/model1.log 2>&1 &
INSTANCE2_PID=$!
echo "Instance 2 deployed on port 8003 using GPU 1"

# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Handle cleanup on termination
trap "kill $INSTANCE1_PID $INSTANCE2_PID" SIGTERM
wait $INSTANCE1_PID $INSTANCE2_PID