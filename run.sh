#!/usr/bin/env bash
# Builds the project and produces the full set of execution artifacts.
#
#   ./run.sh                 full demo: all images, every operation, logs,
#                            CSV timings, CPU verification and PNG previews
#   ./run.sh --flag=value    just build and run the binary with your flags
#
# Logs are written to results/logs, previews to results/previews.
set -euo pipefail
cd "$(dirname "$0")"

BIN=bin/gpu_batch_pipeline
INPUT_DIR=data/input
OUTPUT_DIR=data/output
LOG_DIR=results/logs

make -j"$(nproc)"

if [[ $# -gt 0 ]]; then
  exec "${BIN}" "$@"
fi

if ! compgen -G "${INPUT_DIR}/*.p[gp]m" > /dev/null; then
  echo "No input images found - downloading and converting the SIPI dataset"
  python3 scripts/prepare_data.py --out_dir "${INPUT_DIR}"
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
nvidia-smi | tee "${LOG_DIR}/00_nvidia_smi.log"

echo "=== 1. Full pipeline (equalize -> blur -> sobel) on every image ==="
"${BIN}" --input_dir="${INPUT_DIR}" --output_dir="${OUTPUT_DIR}/pipeline" \
  --operation=pipeline --sigma=1.5 --streams=4 --save_intermediate \
  --verify=20 --csv="${LOG_DIR}/pipeline_timings.csv" \
  2>&1 | tee "${LOG_DIR}/01_pipeline_all_images.log"

for op in equalize blur sobel; do
  echo "=== 2. Single operation: ${op} on every image ==="
  "${BIN}" --input_dir="${INPUT_DIR}" --output_dir="${OUTPUT_DIR}/${op}" \
    --operation="${op}" --sigma=2.5 --streams=4 --verify=5 \
    2>&1 | tee "${LOG_DIR}/02_${op}_all_images.log"
done

echo "=== 3. Binary edge map (threshold) ==="
"${BIN}" --input_dir="${INPUT_DIR}" --output_dir="${OUTPUT_DIR}/edges_binary" \
  --operation=pipeline --threshold=60 --verify=5 \
  2>&1 | tee "${LOG_DIR}/03_pipeline_threshold60.log"

echo "=== 4. Stream-count scaling (benchmark mode: images preloaded) ==="
for streams in 1 2 4 8; do
  "${BIN}" --input_dir="${INPUT_DIR}" --operation=pipeline \
    --streams="${streams}" --benchmark \
    2>&1 | tail -n 6 | sed "s/^/[streams=${streams}] /"
done | tee "${LOG_DIR}/04_stream_scaling.log"

echo "=== 5. Before/after previews ==="
python3 scripts/make_previews.py --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/pipeline" --preview_dir results/previews \
  --count 14
python3 scripts/make_previews.py --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}/edges_binary" \
  --preview_dir results/previews_binary_edges --count 6

echo "Done. Logs: ${LOG_DIR}  Previews: results/previews"
