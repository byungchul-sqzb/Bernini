#!/usr/bin/env bash
# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

# Single-GPU Bernini-R Gradio demo
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

CUDA_DEVICE=${CUDA_DEVICE:-0}
BERNINI_R_CONFIG=${BERNINI_R_CONFIG:-./pretrained_models/Bernini-R-Diffusers}
PORT=${PORT:-7860}
SHARE=${SHARE:-1}

share_args=()
if [[ "$SHARE" == "1" || "$SHARE" == "true" || "$SHARE" == "TRUE" ]]; then
  share_args+=(--share)
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python gradio_demo.py \
  --config "$BERNINI_R_CONFIG" \
  --port "$PORT" \
  "${share_args[@]}"
