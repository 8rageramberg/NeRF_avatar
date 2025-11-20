# Nerfify_2 – Minimal Person NeRF

This subfolder contains a clean slate pipeline for the “person turntable” set
(≈40–50 downscaled frames at ~720×1024). It keeps everything lightweight while
still running entirely on CUDA via PyTorch3D.

## Quick Workflow

1. **Prepare images**
   ```bash
   python scripts/prepare_images.py session_06 --max-size 1024 --clean
   ```
2. **Run COLMAP once**
   ```bash
   bash Nerfify_2/run_colmap.sh data/session_06
   ```
   (Optional: load `colmap_sparse/0` into COLMAP GUI locally, rotate upright,
   export TXT, and copy back to `data/session_06/colmap_text`.)

3. **Generate masks (optional but recommended)**
   ```bash
   python Nerfify_2/masks.py data/session_06/images --out data/session_06/masks --device cuda
   ```

4. **Train** (Baseline or masked)
   ```bash
   python Nerfify_2/train.py \
     --data_root data/session_06 \
     --out runs/session06_baseline_simple \
     --iters 20000 \
     --n_rays 8192 \
     --n_pts 128 \
     --max_depth 12.0
   ```
   With masks:
   ```bash
   python Nerfify_2/train.py \
     --data_root data/session_06 \
     --masks_dir data/session_06/masks \
     --out runs/session06_mask_simple \
     --mask_weight 4.0
   ```

5. **Cluster runs**
   - `Nerfify_2/sbatch_session06_baseline.sbatch`
   - `Nerfify_2/sbatch_session06_mask.sbatch`
   Copy to `/home/brage/D1/project/NeRFify/Nerfify_2/`, then submit with `sbatch`.

## Files

- `run_colmap.sh` – stripped-down COLMAP helper (no alignment tricks).
- `dataset.py` – parses COLMAP text + loads images (keeps the subject centered).
- `nerf_model.py` – harmonic-embedding NeRF from the cow tutorial, tuned for 3D people.
- `train.py` – simple CUDA training loop and preview renderer.
- `masks.py` – DeepLabV3-based mask generator (outputs 1024-high masks).
- `sbatch_session06_baseline.sbatch` / `sbatch_session06_mask.sbatch` – cluster jobs.

Focus is on correctness + speed: raymarching uses PyTorch3D’s CUDA kernels, and
the training loop keeps sampling straightforward so you can iterate quickly.
