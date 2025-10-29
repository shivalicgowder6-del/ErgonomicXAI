"""Import local images into the project for processing.

Place manufacturing or workplace images in a local folder and run this script to copy and validate them
into `data/images/manufacturing/` so the pipeline can process them.

Example (PowerShell):
  python scripts/import_local_images.py --src C:\path\to\my_images --limit 30 --out data/images/manufacturing

The script will skip non-image files and avoid overwriting existing files. It prints a short summary.
"""
import argparse
from pathlib import Path
import shutil
import imghdr
import sys


def is_image_file(path: Path):
    try:
        # imghdr can detect common types; fallback to suffix check
        kind = imghdr.what(str(path))
        if kind:
            return True
        return path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    except Exception:
        return False


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source folder containing images to import')
    parser.add_argument('--out', default='data/images/manufacturing', help='Destination folder in the project')
    parser.add_argument('--limit', type=int, default=0, help='Maximum number of images to import (0 = all)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be copied without making changes')
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists() or not src.is_dir():
        print('Source folder does not exist or is not a directory:', src)
        sys.exit(1)

    out = Path(args.out)
    safe_mkdir(out)

    files = list(src.iterdir())
    imported = 0
    skipped = 0
    for p in files:
        if p.is_file() and is_image_file(p):
            if args.limit and imported >= args.limit:
                break
            dest_name = p.name
            dest = out / dest_name
            # avoid overwrite: if exists, add suffix
            if dest.exists():
                base = dest.stem
                suffix = dest.suffix
                i = 1
                while (out / f"{base}_{i}{suffix}").exists():
                    i += 1
                dest = out / f"{base}_{i}{suffix}"
            if args.dry_run:
                print('Would copy:', p, '->', dest.name)
            else:
                shutil.copy2(p, dest)
            imported += 1
        else:
            skipped += 1

    print(f'Imported {imported} images to {out} (skipped {skipped} non-image files)')


if __name__ == '__main__':
    main()
