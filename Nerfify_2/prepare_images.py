#!/usr/bin/env python3
"""Utility that downscales raw uploads into a NeRFify session folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nerfify.io import safe_image_read

VALID_EXTS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downscale raw uploads into a session images/ folder.")
    parser.add_argument(
        "session",
        help="Session name, e.g. session_01. Images are written to data/<session>/images/",
    )
    parser.add_argument(
        "--src-root",
        default="data/unprocessed",
        type=Path,
        help="Directory that holds raw uploads (default: data/unprocessed).",
    )
    parser.add_argument(
        "--dst-root",
        default="data",
        type=Path,
        help="Root directory for processed sessions (default: data).",
    )
    parser.add_argument(
        "--max-size",
        default=1024,
        type=int,
        help="Resize so the longest edge is at most this many pixels (default: 1024).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the existing images/ directory before writing new files.",
    )
    return parser.parse_args()


def iter_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in VALID_EXTS:
            yield path


try:
    _LANCZOS = Image.Resampling.LANCZOS  # Pillow >= 10
except AttributeError:  # pragma: no cover - older Pillow fallback
    _LANCZOS = Image.LANCZOS


def downscale_image(img: Image.Image, max_size: int) -> Image.Image:
    """Return a copy resized so the longest edge is <= max_size."""

    if max(img.width, img.height) <= max_size:
        return img
    resized = img.copy()
    resized.thumbnail((max_size, max_size), _LANCZOS)
    return resized


def main() -> None:
    args = parse_args()
    src_dir = (args.src_root / args.session).expanduser()
    dst_dir = (args.dst_root / args.session / "images").expanduser()

    if not src_dir.exists():
        raise SystemExit(f"No uploads found at {src_dir}")

    raw_images = list(iter_images(src_dir))
    if not raw_images:
        raise SystemExit(f"No .jpg/.jpeg/.png files found in {src_dir}")

    if dst_dir.exists():
        if args.clean:
            shutil.rmtree(dst_dir)
        elif any(dst_dir.iterdir()):
            raise SystemExit(f"{dst_dir} already contains files. Use --clean to overwrite.")

    dst_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(raw_images, start=1):
        img = safe_image_read(path)
        processed = downscale_image(img, args.max_size)
        out_path = dst_dir / path.name
        processed.save(out_path)
        print(f"[{idx}/{len(raw_images)}] Wrote {out_path.name} ({processed.width}x{processed.height})")

    print(f"Session ready at {dst_dir}")


if __name__ == "__main__":
    main()
