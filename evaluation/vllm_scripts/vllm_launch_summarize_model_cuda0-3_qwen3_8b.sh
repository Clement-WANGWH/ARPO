#!/bin/bash

# Activate the Conda environment
source /root/miniconda3/etc/profile.d/conda.sh
#source /root/miniconda3/envs/evaluation/bin/activate
conda activate evaluation

# Switch to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

# Create log directory
mkdir -p logs

# Model path - all instances use the same model
MODEL_PATH="root/autodl-tmp/Qwen3-8B"
MODEL_NAME="Qwen3-8B"

# Launch Instance 2 - using GPU 1
echo "Starting Instance 2 on GPU 1"
CUDA_VISIBLE_DEVICES=1 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.9 \
    --port 8004 > logs/model1.log 2>&1 &
INSTANCE2_PID=$!
echo "Instance 2 deployed on port 8004 using GPU 1"

# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Gracefully terminate both instances on SIGTERM
trap "kill $INSTANCE2_PID" SIGTERM
wait $INSTANCE2_PID
