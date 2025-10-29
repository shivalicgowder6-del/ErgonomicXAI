"""Small helper: copy person_keypoints_val2017.json to pipeline filename and validate JSON counts.

Run: python scripts/_copy_coco_keypoints_to_pipeline.py
"""
import json
from pathlib import Path

src = Path('data/annotations/annotations_trainval2017/annotations/person_keypoints_val2017.json')
dst = Path('data/annotations/coco_wholebody_val_v1.0.json')

if not src.exists():
    print('Source not found:', src)
    raise SystemExit(1)

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
print('Copied to', dst)

# quick validate
data = json.loads(dst.read_text(encoding='utf-8'))
print('images:', len(data.get('images', [])))
print('annotations:', len(data.get('annotations', [])))
print('example image entries:')
for im in data.get('images', [])[:5]:
    print(im.get('id'), im.get('file_name'))
