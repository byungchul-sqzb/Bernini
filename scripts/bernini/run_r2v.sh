#!/usr/bin/env bash
set -euo pipefail

# Single-GPU Bernini reference-to-video
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

CUDA_DEVICE=${CUDA_DEVICE:-0}
BERNINI_CONFIG=${BERNINI_CONFIG:-ByteDance/Bernini-Diffusers}
CASE_PATH=${CASE_PATH:-assets/testcases/r2v/r2v.json}

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python infer_single_gpu.py \
  --config "$BERNINI_CONFIG" \
  --case "$CASE_PATH" \
  --num_frames 81 \
  --max_image_size 842 \
  --num_inference_steps 40 \
  --flow_shift 5.0 \
  --seed 42 \
  --fps 16 \
  --omega_txt 4 \
  --omega_tgt 1.5 \
  --omega_img 4.5 \
  --omega_vid 1.25 \
  --omega_scale 0.8 \
  --vit_denoising_step 5 \
  --vit_txt_cfg 1.2 \
  --vit_img_cfg 1.0 \
  --guidance_mode vae_txt_vit_wapg
