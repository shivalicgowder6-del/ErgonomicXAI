#!/usr/bin/env python3
"""Download a small subset of COCO-WholeBody validation images referenced
in the COCO-WholeBody annotations JSON and save them to data/images/.

This script downloads the annotation JSON from the original repo and then
fetches a random subset of images from the COCO val2017 image host using the
`file_name` field in the JSON.

Usage:
    python scripts/get_coco_wholebody_subset.py --num 30
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests

# make sure project root is on sys.path so local imports work when scripts are run directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


RAW_JSON_URL = (
    "https://raw.githubusercontent.com/jin-s13/COCO-WholeBody/master/annotations/"
    "coco_wholebody_val_v1.0.json"
)
LOCAL_JSON = Path(__file__).resolve().parents[1] / "data" / "annotations" / "coco_wholebody_val_v1.0.json"

# COCO val image base URL (standard COCO hosting)
COCO_VAL_BASE = "http://images.cocodataset.org/val2017/"


def download(url, dest: Path, chunk_size=8192):
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)


def main(num=30, seed=42):
    out_dir = Path(__file__).resolve().parents[1] / "data" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer a local annotations JSON if it exists (works offline / avoids remote 404s)
    j = None
    if LOCAL_JSON.exists():
        print(f"Loading local annotation JSON from {LOCAL_JSON} ...")
        try:
            with open(LOCAL_JSON, "r", encoding="utf-8") as fh:
                j = json.load(fh)
        except Exception as e:
            print(f"  - Failed to parse local JSON: {e}")

    if j is None:
        try:
            print(f"Fetching annotation JSON from {RAW_JSON_URL} ...")
            r = requests.get(RAW_JSON_URL, timeout=30)
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            print(f"  - Could not load annotation JSON: {e}")

    # If we still don't have a valid annotation JSON, fall back to curated public images
    if j is None:
        print("Falling back to downloading random public person images from Unsplash.")
        # Use the Unsplash Source API to fetch random 'person' images
        unsplash_url = "https://source.unsplash.com/random/800x800/?person"
        chosen = []
        for i in range(min(num, 100)):
            file_name = f"unsplash_person_{i+1:03d}.jpg"
            url = unsplash_url
            dest = out_dir / file_name
            if dest.exists():
                print(f"  - already exists: {file_name}")
                chosen.append({"file_name": file_name})
                continue
            try:
                print(f"  - fetching {file_name} from Unsplash ...", end=" ")
                download(url, dest)
                print("done")
                chosen.append({"file_name": file_name})
            except Exception as e:
                print(f"FAILED ({e})")
        print(f"Downloaded {len(chosen)} images from Unsplash to {out_dir}.")
        return
    images = j.get("images") or j.get("images_info") or []
    if not images:
        raise RuntimeError("No 'images' field found in the annotation JSON")

    random.seed(seed)
    chosen = random.sample(images, min(num, len(images)))

    print(f"Downloading {len(chosen)} images to {out_dir} ...")
    downloaded = 0
    for img in chosen:
        file_name = img.get("file_name") or img.get("file_name_val") or img.get("file")
        if not file_name:
            continue
        url = urljoin(COCO_VAL_BASE, file_name)
        dest = out_dir / file_name
        if dest.exists():
            print(f"  - already exists: {file_name}")
            downloaded += 1
            continue
        try:
            print(f"  - fetching {file_name} ...", end=" ")
            download(url, dest)
            print("done")
            downloaded += 1
        except Exception as e:
            print(f"FAILED ({e})")

    print(f"Downloaded {downloaded}/{len(chosen)} images.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=30, help="number of images to download")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    args = p.parse_args()
    main(num=args.num, seed=args.seed)
