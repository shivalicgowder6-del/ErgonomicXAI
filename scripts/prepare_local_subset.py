#!/usr/bin/env python3
"""Prepare a local image subset by duplicating existing images in `data/images/`.

This is a safe fallback when network downloads fail. It copies existing images with
new names until the requested target count is reached.

Usage:
    python scripts/prepare_local_subset.py --target 30
"""
import argparse
from pathlib import Path
import shutil


def main(target=30):
    root = Path(__file__).resolve().parents[1]
    img_dir = root / "data" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        print("No source images found in data/images/. Please add at least one image.")
        return

    count = len(imgs)
    print(f"Found {count} source images. Preparing up to {target} images by duplicating.")

    idx = 0
    while count < target:
        src = imgs[idx % len(imgs)]
        dst = img_dir / f"dup_{count+1:03d}_{src.name}"
        shutil.copy(src, dst)
        count += 1
        idx += 1
    print(f"Prepared {count} images in {img_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=int, default=30)
    args = p.parse_args()
    main(target=args.target)
