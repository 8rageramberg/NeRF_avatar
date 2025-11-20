#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <session_dir>"
  exit 1
fi

SESSION_DIR="$1"
IMAGES_DIR="${SESSION_DIR}/images"
DATABASE_PATH="${SESSION_DIR}/colmap_simple.db"
SPARSE_DIR="${SESSION_DIR}/colmap_sparse_simple"
TEXT_DIR="${SESSION_DIR}/colmap_text"

if ! command -v colmap >/dev/null 2>&1; then
  echo "COLMAP binary not available on PATH."
  exit 2
fi

mkdir -p "${SPARSE_DIR}"
rm -rf "${TEXT_DIR}"
mkdir -p "${TEXT_DIR}"
rm -f "${DATABASE_PATH}"

log() {
  echo "[nerfify2-colmap] $*"
}

FEATURE_EXTRA_ARGS=(
  "--SiftExtraction.use_gpu=1"
  "--SiftExtraction.max_image_size=4096"
  "--SiftExtraction.max_num_features=20000"
  "--SiftExtraction.peak_threshold=0.006"
  "--SiftExtraction.edge_threshold=10"
)

MATCHER_EXTRA_ARGS=(
  "--SiftMatching.use_gpu=1"
  "--SiftMatching.max_ratio=0.95"
  "--SiftMatching.max_distance=0.9"
  "--SiftMatching.cross_check=1"
)

MAPPER_EXTRA_ARGS=(
  "--Mapper.ba_local_max_num_iterations=30"
  "--Mapper.ba_global_max_num_iterations=50"
  "--Mapper.init_min_tri_angle=2"
  "--Mapper.init_min_num_inliers=50"
  "--Mapper.abs_pose_min_num_inliers=15"
  "--Mapper.abs_pose_max_error=8"
  "--Mapper.min_model_size=3"
  "--Mapper.min_num_matches=12"
  "--Mapper.ignore_watermarks=1"
)

log "Feature extraction"
colmap feature_extractor \
  --database_path "${DATABASE_PATH}" \
  --image_path "${IMAGES_DIR}" \
  "${FEATURE_EXTRA_ARGS[@]}"

log "Feature matching"
colmap exhaustive_matcher \
  --database_path "${DATABASE_PATH}" \
  "${MATCHER_EXTRA_ARGS[@]}"

log "Incremental mapper"
colmap mapper \
  --database_path "${DATABASE_PATH}" \
  --image_path "${IMAGES_DIR}" \
  --output_path "${SPARSE_DIR}" \
  "${MAPPER_EXTRA_ARGS[@]}"

log "Exporting TXT model"
colmap model_converter \
  --input_path "${SPARSE_DIR}/0" \
  --output_path "${TEXT_DIR}" \
  --output_type TXT

echo "[nerfify2-colmap] Finished at ${TEXT_DIR}"

# Count registered images
IMAGES_TXT="${TEXT_DIR}/images.txt"
if [ -f "${IMAGES_TXT}" ]; then
  NUM_REGISTERED=$(grep -v '^#' "${IMAGES_TXT}" | sed -n '1~2p' | wc -l)
  echo "[nerfify2-colmap] Registered images: ${NUM_REGISTERED}"
else
  echo "[nerfify2-colmap] WARNING: images.txt not found"
fi