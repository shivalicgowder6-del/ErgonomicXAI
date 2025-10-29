"""Download images referenced in a COCO JSON into data/images/coco_from_json/.

Usage: python scripts/download_images_from_coco.py --limit 50
"""
import argparse
import json
from pathlib import Path
import requests
import time


def download_images(json_path: Path, out_dir: Path, limit: int = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    j = json.loads(json_path.read_text(encoding='utf-8'))
    images = j.get('images', [])
    print(f'Found {len(images)} image entries in JSON')
    count = 0
    for im in images:
        if limit and count >= limit:
            break
        url = im.get('coco_url') or im.get('url')
        fname = im.get('file_name') or None
        if not url and not fname:
            continue
        if fname:
            target = out_dir / fname
        else:
            from urllib.parse import urlparse
            target = out_dir / Path(urlparse(url).path).name
        if target.exists():
            print('skip exists:', target.name)
            count += 1
            continue
        if not url:
            print('No URL for', fname, 'skipping')
            count += 1
            continue
        try:
            print('downloading:', url)
            r = requests.get(url, timeout=30, stream=True)
            r.raise_for_status()
            with open(target, 'wb') as fh:
                for chunk in r.iter_content(1024 * 32):
                    if chunk:
                        fh.write(chunk)
            time.sleep(0.05)
        except Exception as e:
            print('failed:', url, e)
        count += 1
    print('done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='data/annotations/coco_wholebody_val_v1.0.json')
    parser.add_argument('--out', default='data/images/coco_from_json')
    parser.add_argument('--limit', type=int, default=50)
    args = parser.parse_args()
    json_path = Path(args.json)
    if not json_path.exists():
        print('JSON not found:', json_path)
        raise SystemExit(1)
    download_images(json_path, Path(args.out), args.limit)


if __name__ == '__main__':
    main()
