import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from pytorch3d.renderer import ImplicitRenderer
from PIL import Image as PILImage

from dataset import load_session
from nerf_model import NeuralRadianceField, make_implicit_renderer

LOGGER = logging.getLogger(__name__)


def parse_args():
    # Define CLI arguments for training a NeRF on a turntable dataset.
    parser = argparse.ArgumentParser(description="Tiny NeRF for person turntable.")
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--masks_dir", type=Path)
    parser.add_argument("--device", default="cuda")

    # Training
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_rays", type=int, default=8192)
    parser.add_argument("--n_pts", type=int, default=128)

    # Depth range (slightly tighter defaults; can be overridden on CLI)
    parser.add_argument("--min_depth", type=float, default=0.5)
    parser.add_argument("--max_depth", type=float, default=5.0)

    parser.add_argument("--preview_every", type=int, default=500)

    # Mask weighting (stronger by default)
    parser.add_argument("--mask_weight", type=float, default=8.0)
    return parser.parse_args()


def main():
    # End-to-end training loop: load data, build model, optimize, and save previews.
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    images_np, cameras, masks_np = load_session(Path(args.data_root), device=device, masks_dir=args.masks_dir)
    images = torch.from_numpy(images_np).float().permute(0, 3, 1, 2) / 255.0
    images = images.to(device)

    masks = None
    if masks_np is not None:
        masks = torch.from_numpy(masks_np).unsqueeze(1).float() / 255.0
        masks = masks.to(device)

    # ------------------------------------------------------------------
    # Model + renderer
    # ------------------------------------------------------------------
    field = NeuralRadianceField().to(device)
    renderer_mc, renderer_full = make_implicit_renderer(
        image_height=images.shape[-2],
        image_width=images.shape[-1],
        n_pts_per_ray=args.n_pts,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        n_rays_per_image=args.n_rays,
    )
    optimizer = torch.optim.Adam(field.parameters(), lr=args.lr)

    # Multi-step LR schedule: 1.0 -> 0.3 -> 0.09 at given milestones
    scheduler = MultiStepLR(
        optimizer,
        milestones=[8000, 15000],
        gamma=0.3,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for step in range(1, args.iters + 1):
        idx = torch.randint(0, images.shape[0], (1,), device=device).item()
        gt = images[idx: idx + 1]
        cam = cameras[idx]

        # Sample rays and render
        ray_bundle = renderer_mc.raysampler(cameras=cam)
        rays_density, rays_color = field(ray_bundle)
        rendered = renderer_mc.raymarcher(rays_density, rays_color)[..., :-1]

        # Sample GT at ray locations
        sample_grid = ray_bundle.xys.view(1, 1, -1, 2)
        gt_sampled = (
            F.grid_sample(gt, sample_grid, align_corners=True)
            .permute(0, 2, 3, 1)
            .view_as(rendered)
        )

        # Base color loss
        if masks is None:
            loss = F.mse_loss(rendered, gt_sampled)
        else:
            # Foreground-weighted loss
            mask_sampled = (
                F.grid_sample(masks[idx: idx + 1], sample_grid, align_corners=True)
                .permute(0, 2, 3, 1)
                .view(rendered.shape[0], rendered.shape[1], 1)
            )
            weights = 1.0 + args.mask_weight * mask_sampled
            loss = (weights * (rendered - gt_sampled) ** 2).mean()

        # Light density regularizer (slightly weaker)
        loss = loss + 5e-5 * rays_density.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            LOGGER.info("Iter %05d loss %.6f lr %.6e", step, loss.item(), current_lr)

        if step % args.preview_every == 0 or step == args.iters:
            save_preview(
                renderer_full,
                field,
                cameras[idx],
                args.out / f"preview_{step:05d}.png",
            )

    torch.save({"model": field.state_dict()}, args.out / "final.pt")
    LOGGER.info("Training complete. Checkpoints at %s", args.out)


def save_preview(
    renderer: ImplicitRenderer,
    field: NeuralRadianceField,
    camera,
    out_path: Path,
    n_batches: int = 32,
):
    # Render a full-frame preview from the current model state.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        raysampler = renderer.raysampler
        ray_bundle = raysampler(cameras=camera)
        sigma, color = field.batched_forward(ray_bundle, n_batches=n_batches)
        image = renderer.raymarcher(sigma, color)[..., :-1][0]
        image_np = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        render_cfg = getattr(renderer, "render_cfg", None)
        if render_cfg is not None:
            h = render_cfg["image_height"]
            w = render_cfg["image_width"]
        else:
            h = int(np.sqrt(image_np.shape[0]))
            w = image_np.shape[0] // h

        PILImage.fromarray(image_np.reshape(h, w, 3)).save(out_path)


if __name__ == "__main__":
    main()
