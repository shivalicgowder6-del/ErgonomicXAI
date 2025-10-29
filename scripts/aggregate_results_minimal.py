"""Lightweight aggregator for per-image JSON outputs.

Creates a CSV summary and a simple PNG bar chart of mean part importances.
This script avoids pandas/matplotlib to work around NumPy/matplotlib ABI issues.
"""
from pathlib import Path
import json
import csv
import argparse
from statistics import mean


def aggregate(indir: Path):
    indir = Path(indir)
    out_csv = indir / 'summary.csv'
    jsons = sorted(indir.glob('*.json'))
    rows = []
    for j in jsons:
        try:
            data = json.loads(j.read_text(encoding='utf-8'))
        except Exception:
            continue
        image = data.get('image', j.stem + '.jpg')
        reba = data.get('reba', None)
        parts = data.get('parts', {}) or {}
        trunk = parts.get('trunk', '')
        arms = parts.get('arms', '')
        legs = parts.get('legs', '')
        advice_shap = data.get('advice_shap', '')
        advice_det = data.get('advice_det', '')
        primary_shap = data.get('primary_shap', '')
        primary_det = data.get('primary_det', '')
        advice_conflict = data.get('advice_conflict', False)
        rows.append({
            'image': image, 'reba': reba, 'trunk': trunk, 'arms': arms, 'legs': legs,
            'advice_shap': advice_shap, 'advice_det': advice_det,
            'primary_shap': primary_shap, 'primary_det': primary_det,
            'advice_conflict': advice_conflict
        })

    # write CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        fieldnames = ['image','reba','trunk','arms','legs','advice_shap','advice_det','primary_shap','primary_det','advice_conflict']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote CSV: {out_csv} rows={len(rows)}')

    # try to create a simple PNG summary plot using Pillow if available
    try:
        from PIL import Image, ImageDraw, ImageFont
        # compute means
        trunks = [float(r['trunk']) for r in rows if r['trunk']!='']
        arms = [float(r['arms']) for r in rows if r['arms']!='']
        legs = [float(r['legs']) for r in rows if r['legs']!='']
        avg = {'trunk': mean(trunks) if trunks else 0.0, 'arms': mean(arms) if arms else 0.0, 'legs': mean(legs) if legs else 0.0}

        # image layout
        W, H = 600, 200
        im = Image.new('RGB', (W, H), 'white')
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        parts = ['trunk','arms','legs']
        colors = {'trunk': (76,114,176), 'arms': (221,132,82), 'legs': (85,168,104)}
        gap = 20
        bar_w = 300
        x0 = 150
        y = 40
        for p in parts:
            val = avg.get(p, 0.0)
            draw.text((10, y+4), p, fill='black', font=font)
            draw.rectangle([x0, y, x0 + int(bar_w*val), y+30], fill=colors[p])
            draw.text((x0+bar_w+10, y+4), f'{val:.2f}', fill='black', font=font)
            y += 40

        out_png = indir / 'summary_plot.png'
        im.save(out_png)
        print('Wrote PNG:', out_png)
    except Exception as e:
        print('Pillow not available or plotting failed:', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True, help='Directory with per-image JSONs')
    args = parser.parse_args()
    aggregate(Path(args.indir))


if __name__ == '__main__':
    main()
