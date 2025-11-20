#!/usr/bin/env bash
#SBATCH --job-name=nerfify-smoke
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/brage/D1/project/NeRFify
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euxo pipefail

module purge
module load slurm/21.08.8
module load cuda12.4/toolkit/12.4.1

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)" || true
fi
conda activate /home/brage/miniconda3/envs/nerfify

SESSION=${1:-session_02}
DATA_ROOT="/home/brage/D1/project/NeRFify/data/${SESSION}"
MASK_DIR="/home/brage/D1/project/NeRFify/data/${SESSION}/masks_smoke_cluster"
RUN_DIR="/home/brage/D1/project/NeRFify/runs/${SESSION}_smoke_cluster_${SLURM_JOB_ID}"

bash scripts/run_colmap.sh "$DATA_ROOT"
python -m nerfify.masks "$DATA_ROOT/images" --out "$MASK_DIR" --device cuda
python -m nerfify.train_with_masks \
  --data_root "$DATA_ROOT" \
  --masks_dir "$MASK_DIR" \
  --out "$RUN_DIR" \
  --iters 300 \
  --device cuda \
  --n_rays 512 \
  --n_pts 32 \
  --mask_weight 4.0 \
  --preview_height 128 \
  --preview_width 128 \
  --preview_every 100
python -m nerfify.render_demo \
  --data_root "$DATA_ROOT" \
  --ckpt "$RUN_DIR/final.pt" \
  --out "$RUN_DIR/smoke_preview.mp4" \
  --frames 5 --height 128 --width 128 --device cuda
