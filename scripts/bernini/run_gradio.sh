#!/usr/bin/env bash
set -euo pipefail

# Single-GPU Bernini Gradio demo
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

CUDA_DEVICE=${CUDA_DEVICE:-0}
BERNINI_CONFIG=${BERNINI_CONFIG:-ByteDance/Bernini-Diffusers}
PORT=${PORT:-9500}
SHARE=${SHARE:-1}

share_args=()
if [[ "$SHARE" == "1" || "$SHARE" == "true" || "$SHARE" == "TRUE" ]]; then
  share_args+=(--share)
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python gradio_demo.py \
  --config "$BERNINI_CONFIG" \
  --port "$PORT" \
  "${share_args[@]}"
