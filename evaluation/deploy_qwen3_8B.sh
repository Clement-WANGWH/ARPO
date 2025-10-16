#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
#source /root/miniconda3/envs/evaluation/bin/activate
conda activate sft


export CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0 nohup vllm serve /root/autodl-tmp/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --max-model-len 32768 \
  --tensor_parallel_size 1 \
  --gpu-memory-utilization 0.9 \
  --port 8001