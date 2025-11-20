#!/usr/bin/env bash
#SBATCH --job-name=nerf2-session06-base
#SBATCH --partition=hgx2q
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
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

set +u
source /home/brage/D1/project/NeRFify/.conda/etc/profile.d/conda.sh
conda activate /home/brage/D1/project/NeRFify/.conda/envs/nerfify
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SESSION=/home/brage/D1/project/NeRFify/data/session_06
RUN_DIR="runs/session06_baseline_simple_${SLURM_JOB_ID}"

bash Nerfify_2/run_colmap.sh "${SESSION}"

python Nerfify_2/train.py \
  --data_root "${SESSION}" \
  --out "${RUN_DIR}" \
  --iters 20000 \
  --device cuda \
  --n_rays 8192 \
  --n_pts 128 \
  --max_depth 12.0 \
  --preview_every 2000

TRAIN_STATUS=$?
if [[ ${TRAIN_STATUS} -ne 0 ]]; then
  echo "Training failed with exit code ${TRAIN_STATUS}. Check logs/nerf2-session06-base-${SLURM_JOB_ID}.err." >&2
  exit ${TRAIN_STATUS}
fi
