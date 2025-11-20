# Nerfify_2

Minimal person NeRF pipeline built on PyTorch3D. It is meant for turntable shots
of one person (around 40–50 RGB frames). The code is small, uses CUDA when
available, and tries to keep every step easy to follow.

## What you get
- `run_colmap.sh`: runs COLMAP to make `colmap_text/` camera files.
- `masks.py`: makes foreground masks with DeepLabV3 (person class).
- `dataset.py`: loads images, COLMAP text, and optional masks.
- `nerf_model.py`: tiny NeRF with harmonic embeddings and PyTorch3D renderers.
- `train.py`: training and preview rendering loop.
- `sbatch_session*.sbatch`: sample Slurm jobs for the above scripts.

## Data layout
Each session folder should look like this:
```
session_xx/
  images/        # your RGB frames (jpg/png)
  colmap_text/   # made by run_colmap.sh (cameras.txt, images.txt, points3D.txt)
  masks/         # optional png masks matching file names in images/
```
Images are loaded in sorted order, then re-ordered to match `images.txt` from
COLMAP so cameras and pixels stay aligned.

## Requirements
- Python with PyTorch and PyTorch3D
- torchvision, numpy, pillow, scipy
- COLMAP binary on your PATH (for `run_colmap.sh`)
- CUDA GPU is recommended; CPU is supported but slow

## Quick start
1) Put your frames in `data/session_XX/images` (jpg/png).
2) Build cameras with COLMAP:
```bash
bash Nerfify_2/run_colmap.sh data/session_06
```
3) (Optional) Make masks to weight the person more:
```bash
python Nerfify_2/masks.py data/session_06/images --out data/session_06/masks --device cuda
```
4) Train without masks:
```bash
python Nerfify_2/train.py \
  --data_root data/session_06 \
  --out runs/session06_baseline \
  --iters 20000 \
  --n_rays 8192 \
  --n_pts 128 \
  --max_depth 12.0
```
   Or train with masks (stronger foreground weight by default):
```bash
python Nerfify_2/train.py \
  --data_root data/session_06 \
  --masks_dir data/session_06/masks \
  --out runs/session06_mask \
  --mask_weight 8.0
```
5) Look for `preview_XXXXX.png` snapshots and `final.pt` in your `--out` folder.

## Training options (main ones)
- `--data_root`: session folder containing `images/` and `colmap_text/` (required)
- `--out`: where previews and `final.pt` are saved (required)
- `--masks_dir`: folder with png masks (optional)
- `--device`: `cuda` (default) or `cpu`
- `--iters`: training steps (default 20000)
- `--n_rays`: rays per iteration (default 8192)
- `--n_pts`: samples per ray (default 128)
- `--min_depth` / `--max_depth`: depth range in meters (defaults 0.5 / 5.0)
- `--mask_weight`: extra weight on masked pixels when masks are used (default 8.0)
- `--preview_every`: save a preview every N steps (default 500)

The renderer uses Monte Carlo sampling during training and a full-frame render
for previews. Density has a light regularizer to keep values stable.

## Mask generator
- Uses DeepLabV3-ResNet50 (`PERSON_CLASS = 15`).
- Default threshold is `0.5`; change with `--threshold`.
- Resizes output to the input image size and fills small holes.

## COLMAP helper
`run_colmap.sh` expects `session_xx/images` to exist. It creates:
- `session_xx/colmap_simple.db` (COLMAP database)
- `session_xx/colmap_sparse_simple/` (sparse model)
- `session_xx/colmap_text/` (TXT export used by the loader)
It enables GPU SIFT and simple exhaustive matching. The script prints how many
images were registered when done.

## Cluster jobs
Sample Slurm scripts live here:
- `sbatch_session06_baseline.sbatch`
- `sbatch_session06_mask.sbatch`
- `sbatch_session11_mask.sbatch`
- `sbatch_session12_baseline.sbatch`
They show how to run COLMAP, optional masking, and training on a cluster. The
`sbatch_session12_baseline.sbatch` file includes extra `--lambda_*` flags that
are not defined in `train.py`; remove those flags before running it.

## Appendix: Unused lambda flags
An earlier experiment added extra density regularizer flags (`--lambda_bg_density`,
`--lambda_fg_density`, `--lambda_global_density`) in `sbatch_session12_baseline.sbatch`.
We did not integrate these into `train.py`, and the experiment was dropped. The current
code uses only the built-in light density penalty and `--mask_weight`. If you try to run
the script with those lambda flags, argparse will error on unrecognized arguments.
