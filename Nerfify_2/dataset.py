from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras


def load_images(images_dir: Path) -> Tuple[List[Path], np.ndarray]:
    # Load RGB images from a directory and return paths plus stacked numpy array.
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    imgs = [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]
    return paths, np.stack(imgs)


def parse_colmap(text_dir: Path) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    # Parse COLMAP text files into camera and image dictionaries.
    cameras = {}
    images = {}
    with (text_dir / "cameras.txt").open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = map(int, parts[2:4])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    with (text_dir / "images.txt").open() as fh:
        lines = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    idx = 0
    while idx < len(lines):
        header = lines[idx].split()
        image_id = int(header[0])
        qvec = tuple(float(v) for v in header[1:5])
        tvec = tuple(float(v) for v in header[5:8])
        cam_id = int(header[8])
        name = " ".join(header[9:])
        images[image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name}
        idx += 2
    return cameras, images


def quaternion_to_rotation(qvec: Tuple[float, float, float, float]) -> torch.Tensor:
    # Convert COLMAP quaternion into a rotation matrix.
    qw, qx, qy, qz = qvec
    return torch.tensor(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=torch.float32,
    )


def _intrinsics(cam: dict) -> Tuple[float, float, float, float]:
    # Extract fx, fy, cx, cy from the COLMAP camera model.
    params = cam["params"]
    model = cam["model"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        f = params[0]
        cx = params[1]
        cy = params[2]
        return f, f, cx, cy
    if model == "PINHOLE":
        return params[0], params[1], params[2], params[3]
    if model == "OPENCV":
        return params[0], params[1], params[2], params[3]
    raise ValueError(f"Unsupported camera model {model}")


def build_cameras(
    cameras_dict: Dict[int, dict],
    images_dict: Dict[int, dict],
    device: torch.device,
) -> PerspectiveCameras:
    # Build PyTorch3D PerspectiveCameras from COLMAP metadata.
    R_list = []
    T_list = []
    focal_list = []
    pp_list = []
    cv_to_gl = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float32,
    )
    for image_id in sorted(images_dict):
        image = images_dict[image_id]
        cam = cameras_dict[image["camera_id"]]
        width = cam["width"]
        height = cam["height"]
        fx, fy, cx, cy = _intrinsics(cam)

        # COLMAP gives world->camera (OpenCV). Convert to camera->world.
        R_wc_cv = quaternion_to_rotation(image["qvec"])
        t_wc_cv = torch.tensor(image["tvec"], dtype=torch.float32)
        R_cw_cv = R_wc_cv.transpose(0, 1)
        t_cw_cv = (-R_wc_cv.transpose(0, 1) @ t_wc_cv.view(3, 1)).view(-1)

        # Convert orientation to PyTorch3D/OpenGL convention.
        R_gl = cv_to_gl @ R_cw_cv
        t_gl = (cv_to_gl @ t_cw_cv.view(3, 1)).view(-1)

        # Normalize intrinsics to NDC.
        fx_ndc = 2.0 * fx / width
        fy_ndc = 2.0 * fy / height
        cx_ndc = (2.0 * cx / width) - 1.0
        cy_ndc = (2.0 * cy / height) - 1.0

        R_list.append(R_gl)
        T_list.append(t_gl)
        focal_list.append(torch.tensor([fx_ndc, fy_ndc], dtype=torch.float32))
        pp_list.append(torch.tensor([cx_ndc, cy_ndc], dtype=torch.float32))

    return PerspectiveCameras(
        R=torch.stack(R_list).to(device),
        T=torch.stack(T_list).to(device),
        focal_length=torch.stack(focal_list).to(device),
        principal_point=torch.stack(pp_list).to(device),
        device=device,
    )


def load_session(path: Path, device: torch.device, masks_dir: Optional[Path] = None):
    # Load a COLMAP session: images, cameras, and optional foreground masks.
    path = Path(path)
    images_dir = path / "images"
    colmap_dir = path / "colmap_text"
    paths, imgs = load_images(images_dir)
    cameras_dict, images_dict = parse_colmap(colmap_dir)
    cameras = build_cameras(cameras_dict, images_dict, device)
    name_to_idx = {p.name: i for i, p in enumerate(paths)}
    ordered = []
    ordered_masks = []
    for image_id in sorted(images_dict):
        name = images_dict[image_id]["name"]
        ordered.append(imgs[name_to_idx[name]])
        if masks_dir:
            mask_path = masks_dir / Path(name).with_suffix(".png").name
            mask_img = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            ordered_masks.append(mask_img)
    masks_np = np.stack(ordered_masks) if ordered_masks else None
    return np.stack(ordered), cameras, masks_np
