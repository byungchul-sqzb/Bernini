#!/usr/bin/env bash
set -euo pipefail

# Run the six Bernini single-GPU example cases sequentially without overwriting
# the checked-in *_out reference files. Each case writes its own output, log,
# model-run e2e time, and peak VRAM under OUT_DIR.
#
# model_e2e_seconds / peak_vram_mib are emitted by infer_single_gpu.py around
# the pipeline execution call after model loading/initialization has completed.

export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

CUDA_DEVICE=${CUDA_DEVICE:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
BERNINI_CONFIG=${BERNINI_CONFIG:-ByteDance/Bernini-Diffusers}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-outputs/bernini_single_gpu_${RUN_ID}}

mkdir -p "$OUT_DIR" "$OUT_DIR/logs" "$OUT_DIR/cases"
METRICS_CSV="$OUT_DIR/metrics.csv"
printf 'case,task,case_file,output,status,model_e2e_seconds,peak_vram_mib,log\n' > "$METRICS_CSV"

# Negative prompt used by the standalone v2v launcher.
NEG_PROMPT=${NEG_PROMPT:-"vivid tones, overexposed, static, blurry details, subtitles, style, artwork, painting, image, motionless, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, still frame, cluttered background, three legs, too many people in the background, walking backwards"}

make_case() {
  local src=$1
  local dst=$2
  local output=$3
  "$PYTHON_BIN" - "$src" "$dst" "$output" <<'PY'
import json
import sys
from pathlib import Path

src, dst, output = map(Path, sys.argv[1:])
case = json.loads(src.read_text())
case["output"] = str(output)
dst.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
PY
}

read_model_metrics() {
  local metrics_jsonl=$1
  "$PYTHON_BIN" - "$metrics_jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists() or not path.read_text().strip():
    print("NA,NA")
    raise SystemExit
record = json.loads(path.read_text().splitlines()[-1])
print(f"{record.get('model_e2e_seconds', 'NA')},{record.get('peak_vram_mib', 'NA')}")
PY
}

append_metric() {
  local case_name=$1 task=$2 case_file=$3 output=$4 status=$5 elapsed=$6 peak=$7 log=$8
  "$PYTHON_BIN" - "$METRICS_CSV" "$case_name" "$task" "$case_file" "$output" "$status" "$elapsed" "$peak" "$log" <<'PY'
import csv
import sys

csv_path, *row = sys.argv[1:]
with open(csv_path, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY
}

FAILED=0

run_case_or_continue() {
  if ! run_case "$@"; then
    FAILED=$((FAILED + 1))
  fi
}

run_case() {
  local case_name=$1 task=$2 src_case=$3 out_name=$4
  shift 4
  local -a args=("$@")

  local out_path="$OUT_DIR/$out_name"
  local run_case_path="$OUT_DIR/cases/${case_name}.json"
  local log_path="$OUT_DIR/logs/${case_name}.log"
  local model_metrics_jsonl="$OUT_DIR/logs/${case_name}.model_metrics.jsonl"

  make_case "$src_case" "$run_case_path" "$out_path"
  : > "$model_metrics_jsonl"

  echo "[$(date '+%F %T')] START $case_name -> $out_path" | tee "$log_path"

  local status elapsed peak
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" infer_single_gpu.py \
    --config "$BERNINI_CONFIG" \
    --case "$run_case_path" \
    --metrics_jsonl "$model_metrics_jsonl" \
    "${args[@]}" >>"$log_path" 2>&1
  status=$?
  set -e

  IFS=, read -r elapsed peak < <(read_model_metrics "$model_metrics_jsonl")

  if [[ "$status" -eq 0 ]]; then
    echo "[$(date '+%F %T')] PASS $case_name model_e2e=${elapsed}s peak_vram=${peak}MiB" | tee -a "$log_path"
    append_metric "$case_name" "$task" "$run_case_path" "$out_path" "PASS" "$elapsed" "$peak" "$log_path"
  else
    echo "[$(date '+%F %T')] FAIL $case_name status=$status model_e2e=${elapsed}s peak_vram=${peak}MiB" | tee -a "$log_path"
    append_metric "$case_name" "$task" "$run_case_path" "$out_path" "FAIL:$status" "$elapsed" "$peak" "$log_path"
    return "$status"
  fi
}

# Keep these arguments aligned with the per-task single-GPU launchers.
run_case_or_continue t2i t2i assets/testcases/t2i/t2i.json t2i_${RUN_ID}.png \
  --num_frames 1 --max_image_size 842 --height 512 --width 512 \
  --num_inference_steps 50 --flow_shift 5.0 --seed 42 --fps 16 \
  --omega_txt 4 --omega_tgt 0.5 --omega_img 1 --omega_vid 1 --omega_scale 1 \
  --vit_denoising_step 5 --vit_txt_cfg 1.2 --vit_img_cfg 1.0 \
  --guidance_mode vae_txt_vit_wapg

run_case_or_continue i2i i2i assets/testcases/i2i/i2i.json i2i_${RUN_ID}.png \
  --num_frames 1 --max_image_size 842 --height 512 --width 512 \
  --num_inference_steps 40 --flow_shift 5.0 --seed 42 --fps 16 \
  --omega_txt 4 --omega_tgt 0.5 --omega_img 1.25 --omega_vid 1.25 --omega_scale 0.75 \
  --vit_denoising_step 5 --vit_txt_cfg 1.2 --vit_img_cfg 1.0 \
  --guidance_mode vae_txt_vit_wapg

run_case_or_continue t2v t2v assets/testcases/t2v/t2v.json t2v_${RUN_ID}.mp4 \
  --num_frames 81 --max_image_size 842 --height 480 --width 848 \
  --num_inference_steps 50 --flow_shift 5.0 --seed 42 --fps 16 \
  --omega_txt 4 --omega_tgt 0.5 --omega_img 1 --omega_vid 1 --omega_scale 1 \
  --vit_denoising_step 5 --vit_txt_cfg 1.2 --vit_img_cfg 1.0 \
  --guidance_mode vae_txt_vit_wapg

run_case_or_continue v2v v2v assets/testcases/v2v/v2v_case1.json v2v_case1_${RUN_ID}.mp4 \
  --use_unipc --use_src_tgt_id \
  --num_frames 81 --max_image_size 848 --height 0 --width 0 \
  --num_inference_steps 40 --flow_shift 5.0 --seed 42 --fps 16 \
  --omega_txt 4 --omega_tgt 0.5 --omega_img 1.25 --omega_vid 1.25 --omega_scale 0.75 \
  --vit_denoising_step 5 --vit_txt_cfg 1.2 --vit_img_cfg 1.0 \
  --guidance_mode vae_txt_vit_wapg \
  --system_prompt "You are a helpful assistant specialized in video editing." \
  --neg_prompt "$NEG_PROMPT"

run_case_or_continue rv2v rv2v assets/testcases/rv2v/rv2v_case1.json rv2v_case1_${RUN_ID}.mp4 \
  --num_frames 81 --max_image_size 848 --height 0 --width 0 \
  --num_inference_steps 40 --flow_shift 5.0 --seed 42 --fps 16 \
  --omega_txt 4 --omega_tgt 1.5 --omega_img 3 --omega_vid 0.75 --omega_scale 0.75 \
  --vit_denoising_step 5 --vit_txt_cfg 1.2 --vit_img_cfg 1.0 \
  --guidance_mode rv2v_wapg

run_case_or_continue r2v r2v assets/testcases/r2v/r2v.json r2v_${RUN_ID}.mp4 \
  --num_frames 81 --max_image_size 842 \
  --num_inference_steps 40 --flow_shift 5.0 --seed 42 --fps 16 \
  --omega_txt 4 --omega_tgt 1.5 --omega_img 4.5 --omega_vid 1.25 --omega_scale 0.8 \
  --vit_denoising_step 5 --vit_txt_cfg 1.2 --vit_img_cfg 1.0 \
  --guidance_mode vae_txt_vit_wapg

if [[ "$FAILED" -eq 0 ]]; then
  echo "All cases completed."
else
  echo "$FAILED case(s) failed. See per-case logs and $METRICS_CSV." >&2
fi
echo "Outputs: $OUT_DIR"
echo "Metrics: $METRICS_CSV"
exit "$FAILED"
