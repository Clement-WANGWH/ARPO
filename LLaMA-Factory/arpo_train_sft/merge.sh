#!/bin/bash
set -euo pipefail

#============== 环境/并行设置（与训练脚本风格一致） ==============#
export CUDA_VISIBLE_DEVICES=0,1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$(dirname "$SCRIPT_DIR")/src:${PYTHONPATH:-}"
export WANDB_DISABLED=true

#============== 路径（按需修改） ==============#
BASE_MODEL="/root/autodl-tmp/Qwen2.5-7B-Instruct"                             
ADAPTER_DIR="/root/autodl-tmp/checkpoint/Qwen2.5-7B-Instruct"
MERGED_DIR="/root/autodl-tmp/Qwen2.5-7B-Instruct-sft"

# 创建输出目录
mkdir -p "${MERGED_DIR}"

#============== 执行导出（合并） ==============#
# 方式 1：直接参数方式（最直观）
python ../src/llamafactory/merger.py \
  --model_name_or_path "${BASE_MODEL}" \
  --adapter_name_or_path "${ADAPTER_DIR}" \
  --template "qwen" \
  --finetuning_type "lora" \
  --export_dir "${MERGED_DIR}" \
  --export_legacy_format False

# 方式 2：如你更偏好 YAML（与训练时使用 YAML 的风格一致）
# python ../src/export_model.py yaml/merge.yaml

echo "✅ Done. Merged model saved to: ${MERGED_DIR}"
