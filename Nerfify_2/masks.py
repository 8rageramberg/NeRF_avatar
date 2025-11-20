#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

LOGGER = logging.getLogger(__name__)
PERSON_CLASS = 15
_MODEL = None
_WEIGHTS = DeepLabV3_ResNet50_Weights.DEFAULT


def _load_model(device: torch.device):
    # Lazy-load the DeepLabV3 model onto the requested device.
    global _MODEL
    if _MODEL is None:
        LOGGER.info("Loading DeepLabV3 weights")
        _MODEL = deeplabv3_resnet50(weights=_WEIGHTS).to(device)
        _MODEL.eval()
    return _MODEL


def segment(image: Image.Image, device: str, threshold: float) -> Image.Image:
    # Run person segmentation on a PIL image and return a binary mask.
    torch_device = torch.device(device)
    model = _load_model(torch_device)
    preprocess = _WEIGHTS.transforms()
    with torch.no_grad():
        input_tensor = preprocess(image).unsqueeze(0).to(torch_device)
        logits = model(input_tensor)["out"][0]
        probs = logits.softmax(dim=0)
        mask = probs[PERSON_CLASS] > threshold
    mask_np = ndimage.binary_fill_holes(mask.squeeze().cpu().numpy()).astype(np.uint8) * 255
    return Image.fromarray(mask_np, mode="L").resize(image.size, Image.NEAREST)


def main():
    # CLI entry point for generating masks for all images in a directory.
    parser = argparse.ArgumentParser(description="Generate person masks.")
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.out.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(args.images_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        LOGGER.info("Segmenting %s", img_path.name)
        img = Image.open(img_path).convert("RGB")
        mask = segment(img, args.device, args.threshold)
        mask.save(args.out / img_path.with_suffix(".png").name)
    LOGGER.info("Masks saved to %s", args.out)


if __name__ == "__main__":
    main()
