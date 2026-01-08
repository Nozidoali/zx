import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import validate_file_exists, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='artifacts/base.csv')
    parser.add_argument('--gnn-sup', type=str, default='artifacts/gnn_sup.csv')
    parser.add_argument('--gnn-rl', type=str, default='artifacts/gnn_rl.csv')
    parser.add_argument('--output', type=str, default='artifacts/cmp.csv')
    parser.add_argument('--plot', type=str, default='artifacts/cmp.png')
    args = parser.parse_args()

    validate_file_exists(args.base, "Baseline CSV")

    base_df = pd.read_csv(args.base)

    dfs = {'baseline': base_df}
    if Path(args.gnn_sup).exists():
        dfs['gnn_sup'] = pd.read_csv(args.gnn_sup)
    if Path(args.gnn_rl).exists():
        dfs['gnn_rl'] = pd.read_csv(args.gnn_rl)

    ensure_dir(str(Path(args.output).parent))

    comparison = []
    for bench in base_df['benchmark']:
        row = {'benchmark': bench}
        for name, df in dfs.items():
            bench_row = df[df['benchmark'] == bench]
            if not bench_row.empty:
                row[f'{name}_cnot'] = int(bench_row['cnot_count'].iloc[0])
                row[f'{name}_t'] = int(bench_row['t_count'].iloc[0])
                row[f'{name}_total'] = int(bench_row['total_gates'].iloc[0])
        comparison.append(row)

    cmp_df = pd.DataFrame(comparison)
    cmp_df.to_csv(args.output, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['cnot', 't', 'total']
    titles = ['CNOT Count', 'T Count', 'Total Gates']

    for ax, metric, title in zip(axes, metrics, titles):
        cols = [c for c in cmp_df.columns if metric in c]
        data = cmp_df[cols].sum()
        labels = [c.replace(f'_{metric}', '') for c in cols]
        ax.bar(labels, data)
        ax.set_title(title)
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(args.plot)

    validate_file_exists(args.output, "Comparison CSV")
    validate_file_exists(args.plot, "Plot")

    print(f"Comparison complete: saved to {args.output} and {args.plot}")
    sys.exit(0)


if __name__ == '__main__':
    main()
