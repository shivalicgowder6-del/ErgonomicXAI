"""Create a COCO JSON containing only images that have person annotations (category_id == 1).

Run: python scripts/get_coco_person_subset.py --in data/annotations/coco_wholebody_val_v1.0.json --out data/annotations/coco_person_val.json
"""
import argparse
import json
from pathlib import Path


def filter_persons(in_file: Path, out_file: Path):
    j = json.loads(in_file.read_text(encoding='utf-8'))
    images = {im['id']: im for im in j.get('images', [])}
    anns = j.get('annotations', [])
    # category_id for person is usually 1 in COCO
    person_image_ids = set(a['image_id'] for a in anns if a.get('category_id') in (1, 'person'))
    filtered_images = [images[i] for i in person_image_ids if i in images]
    # filter annotations to only those for persons
    filtered_anns = [a for a in anns if a.get('image_id') in person_image_ids and a.get('category_id') in (1, 'person')]

    out = {
        'info': j.get('info', {}),
        'licenses': j.get('licenses', []),
        'images': filtered_images,
        'annotations': filtered_anns,
        'categories': [c for c in j.get('categories', []) if c.get('id') in (1, 'person') or c.get('name') == 'person']
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out))
    print('Wrote person-only JSON:', out_file, 'images:', len(filtered_images), 'annotations:', len(filtered_anns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', default='data/annotations/coco_wholebody_val_v1.0.json')
    parser.add_argument('--out', dest='outfile', default='data/annotations/coco_person_val.json')
    args = parser.parse_args()
    infile = Path(args.infile)
    outfile = Path(args.outfile)
    if not infile.exists():
        print('Input JSON not found:', infile)
        raise SystemExit(1)
    filter_persons(infile, outfile)


if __name__ == '__main__':
    main()
