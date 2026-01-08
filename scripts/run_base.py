#!/usr/bin/env python3
"""Run PyZX baseline extraction on benchmarks."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bench import get_benchmark_list
from src.pzx import run_pyzx_baseline
from src.utils import set_seed, ensure_dir, validate_file_exists, log_verbose
from src.visualize import save_all_benchmarks


def main():
    parser = argparse.ArgumentParser(description='Run baseline extraction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='artifacts/base.csv', help='Output CSV path')
    parser.add_argument('--save-diagrams', action='store_true', help='Save benchmark diagrams')
    parser.add_argument('--diagram-dir', type=str, default='artifacts/benchmarks', help='Diagram output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    benchmarks = get_benchmark_list()
    log_verbose(f"Running baseline on {len(benchmarks)} benchmarks", args.verbose)
    
    if args.save_diagrams:
        log_verbose(f"Saving benchmark diagrams to {args.diagram_dir}/", args.verbose)
        ensure_dir(args.diagram_dir)
        save_all_benchmarks(benchmarks, args.diagram_dir)
    
    ensure_dir(str(Path(args.output).parent))
    
    results = []
    for bench_name, graph in benchmarks:
        log_verbose(f"Processing {bench_name}...", args.verbose)
        
        circuit, metrics = run_pyzx_baseline(graph, args.verbose)
        
        if 'error' in metrics:
            print(f"ERROR: Baseline failed on {bench_name}: {metrics['error']}", file=sys.stderr)
            sys.exit(1)
        
        results.append({
            'benchmark': bench_name,
            'cnot_count': metrics['cnot_count'],
            't_count': metrics['t_count'],
            'total_gates': metrics['total_gates'],
            'time': metrics['time'],
        })
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['benchmark', 'cnot_count', 't_count', 'total_gates', 'time'])
        writer.writeheader()
        writer.writerows(results)
    
    log_verbose(f"Results saved to {args.output}", args.verbose)
    
    validate_file_exists(args.output, "Output file")
    
    total_cnot = sum(r['cnot_count'] for r in results)
    total_t = sum(r['t_count'] for r in results)
    total_gates = sum(r['total_gates'] for r in results)
    print(f"Baseline complete: {len(results)} benchmarks, "
          f"total CNOT={total_cnot}, T={total_t}, gates={total_gates}")
    
    sys.exit(0)


if __name__ == '__main__':
    main()

