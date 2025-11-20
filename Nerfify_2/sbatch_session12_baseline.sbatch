#!/usr/bin/env bash
#SBATCH --job-name=nerf2-session12-baseline
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/brage/D1/project/NeRFify
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euxo pipefail

mkdir -p logs runs

module purge
module load slurm/21.08.8
module load cuda12.4/toolkit/12.4.1

# Activate conda
set +u
source /home/brage/D1/project/NeRFify/.conda/etc/profile.d/conda.sh
conda activate /home/brage/D1/project/NeRFify/.conda/envs/nerfify
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SESSION=/home/brage/D1/project/NeRFify/data/session_12
MASKS_DIR="${SESSION}/masks_simple"
RUN_DIR="runs/session12_baseline_${SLURM_JOB_ID}"

# -------------------------------
# Run COLMAP first
# -------------------------------
bash Nerfify_2/run_colmap.sh "${SESSION}"

# -------------------------------
# Run improved NeRF training
# -------------------------------
python Nerfify_2/train.py \
  --data_root "${SESSION}" \
  --masks_dir "${MASKS_DIR}" \
  --out "${RUN_DIR}" \
  --iters 30000 \
  --device cuda \
  --n_rays 8192 \
  --n_pts 128 \
  --min_depth 0.5 \
  --max_depth 5.0 \
  --mask_weight 8.0 \
  --lambda_bg_density 1e-3 \
  --lambda_fg_density 2e-4 \
  --lambda_global_density 5e-5 \
  --preview_every 300

TRAIN_STATUS=$?

if [[ ${TRAIN_STATUS} -ne 0 ]]; then
  echo "Training failed with exit code ${TRAIN_STATUS}. Check logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2
  exit ${TRAIN_STATUS}
fi

echo "Training finished successfully: ${RUN_DIR}"