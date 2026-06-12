#!/usr/bin/env bash
set -euo pipefail

# Single-GPU Bernini video editing
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

CUDA_DEVICE=${CUDA_DEVICE:-0}
BERNINI_CONFIG=${BERNINI_CONFIG:-ByteDance/Bernini-Diffusers}
CASE_PATH=${CASE_PATH:-assets/testcases/v2v/v2v_case1.json}
NEG_PROMPT=${NEG_PROMPT:-"vivid tones, overexposed, static, blurry details, subtitles, style, artwork, painting, image, motionless, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, still frame, cluttered background, three legs, too many people in the background, walking backwards"}

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python infer_single_gpu.py \
  --config "$BERNINI_CONFIG" \
  --case "$CASE_PATH" \
  --use_unipc \
  --use_src_tgt_id \
  --num_frames 81 \
  --max_image_size 848 \
  --height 0 \
  --width 0 \
  --num_inference_steps 40 \
  --flow_shift 5.0 \
  --seed 42 \
  --fps 16 \
  --omega_txt 4 \
  --omega_tgt 0.5 \
  --omega_img 1.25 \
  --omega_vid 1.25 \
  --omega_scale 0.75 \
  --vit_denoising_step 5 \
  --vit_txt_cfg 1.2 \
  --vit_img_cfg 1.0 \
  --guidance_mode vae_txt_vit_wapg \
  --system_prompt "You are a helpful assistant specialized in video editing." \
  --neg_prompt "$NEG_PROMPT"
