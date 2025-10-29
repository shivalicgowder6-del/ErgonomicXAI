#!/usr/bin/env python3
"""Aggregate per-image JSON result files into one CSV and create a short summary plot.

Writes:
  - validation_results/run_subset_local/summary.csv
  - validation_results/run_subset_local/summary_plot.png

Usage:
  python scripts/aggregate_results.py --indir validation_results/run_subset_local
"""
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(indir: Path):
    p = Path(indir)
    files = sorted(p.glob('*.json'))
    if not files:
        print(f'No JSON files found in {p}')
        return

    rows = []
    for f in files:
        try:
            j = json.loads(f.read_text())
            row = {
                'image': j.get('image'),
                'reba': j.get('reba'),
                'trunk': j.get('parts', {}).get('trunk'),
                'arms': j.get('parts', {}).get('arms'),
                'legs': j.get('parts', {}).get('legs'),
                'advice': j.get('advice')
            }
            rows.append(row)
        except Exception as e:
            print(f'  - skipping {f.name}: {e}')

    df = pd.DataFrame(rows)
    out_csv = p / 'summary.csv'
    df.to_csv(out_csv, index=False)
    print(f'Wrote CSV: {out_csv}')

    # summary plot: REBA histogram + mean parts importance
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].hist(df['reba'].dropna(), bins=range(int(df['reba'].min())-1, int(df['reba'].max())+2), color='#4c72b0')
    axes[0].set_title('REBA distribution')
    axes[0].set_xlabel('REBA')

    mean_parts = df[['trunk','arms','legs']].mean()
    axes[1].bar(mean_parts.index, mean_parts.values, color=['#4c72b0','#dd8452','#55a868'])
    axes[1].set_ylim(0,1)
    axes[1].set_title('Mean parts importance')

    plt.tight_layout()
    out_png = p / 'summary_plot.png'
    fig.savefig(out_png)
    print(f'Wrote plot: {out_png}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='validation_results/run_subset_local')
    args = parser.parse_args()
    main(Path(args.indir))
