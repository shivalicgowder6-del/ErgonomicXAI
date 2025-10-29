"""
Collect up to N processed examples (JSON + viz PNG + parts PNG + original image)
into a submission_examples/ folder and produce a README.md.

Usage:
    python scripts/collect_submission_examples.py --indir validation_results --out submission_examples --n 30
"""
import argparse
from pathlib import Path
import shutil
import json


def collect(indir: Path, outdir: Path, n: int = 30):
    indir = Path(indir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # find JSON files under validation_results
    json_paths = sorted(indir.rglob('*.json'))
    # exclude summary.csv named jsons
    json_paths = [p for p in json_paths if p.name.lower() != 'summary.csv']

    stems = []
    for p in json_paths:
        stem = p.stem
        if stem not in stems:
            stems.append(stem)
        if len(stems) >= n:
            break

    copied = 0
    for stem in stems:
        target = outdir / stem
        target.mkdir(exist_ok=True)
        # locate json
        jp = next(indir.rglob(f'{stem}.json'), None)
        if jp:
            shutil.copy2(jp, target / jp.name)
        # viz
        vz = next(indir.rglob(f'{stem}_viz.png'), None)
        if vz:
            shutil.copy2(vz, target / vz.name)
        # parts
        pr = next(indir.rglob(f'{stem}_parts.png'), None)
        if pr:
            shutil.copy2(pr, target / pr.name)
        # original image: search data/ for matching stem
        img = None
        data_dir = Path(__file__).resolve().parents[1] / 'data'
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            candidate = next(data_dir.rglob(f'{stem}{ext[1:]}'), None)
            if candidate:
                img = candidate
                break
        # fallback: any image file containing stem
        if img is None:
            for f in data_dir.rglob('*'):
                if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png') and stem in f.name:
                    img = f
                    break
        if img:
            shutil.copy2(img, target / img.name)
        copied += 1

    # copy a summary.csv if one exists under validation_results
    summary = next(indir.rglob('summary.csv'), None)
    if summary:
        shutil.copy2(summary, outdir / summary.name)

    # create README
    readme = outdir / 'README.md'
    txt = (
        '# Submission examples for ErgonomicXAI\n\n'
        'This folder contains a curated set of processed examples (JSON + visualization PNG + original image)\n'
        f'Collected from: {indir}\n\n'
        'How to inspect:\n'
        '- Each subfolder is named by the image stem and contains:\n'
        '  - <stem>.json  (fields: image, reba, parts, advice_shap, advice_det, primary_shap, primary_det, advice_conflict)\n'
        '  - <stem>_viz.png (visualization with skeleton + SHAP barchart)\n'
        '  - <stem>_parts.png (small parts bar chart)\n'
        '  - original image file (jpg/png)\n\n'
        'Open apps/streamlit_viewer.py to browse full results or inspect these files directly.\n'
    )
    readme.write_text(txt, encoding='utf-8')
    print(f'Collected {copied} examples into {outdir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='validation_results', help='Root results directory')
    parser.add_argument('--out', default='submission_examples', help='Output submission folder')
    parser.add_argument('--n', type=int, default=30, help='Number of examples to collect')
    args = parser.parse_args()
    collect(Path(args.indir), Path(args.out), args.n)
