import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn import GNNPolicy
from src.bench import get_benchmark_list
from src.infer import batch_inference, greedy_extract
from src.qasm import circuit_to_qasm, validate_qasm
from src.met import count_gates, clifford_t_decompose
from src.utils import set_seed, ensure_dir, validate_file_exists, log_verbose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output', type=str, default='artifacts/gnn.csv')
    parser.add_argument('--qasm-dir', type=str, default='artifacts/qasm')
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--kind-dim', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    validate_file_exists(args.ckpt, "Checkpoint")
    ensure_dir(str(Path(args.output).parent))
    ensure_dir(args.qasm_dir)

    model = GNNPolicy(kind_dim=args.kind_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    model.load_state_dict(torch.load(args.ckpt))

    benchmarks = get_benchmark_list()

    if args.greedy:
        results = [(name, *greedy_extract(model, graph, verbose=args.verbose))
                   for name, graph in benchmarks]
    else:
        results = batch_inference(model, benchmarks, verbose=args.verbose)

    csv_results = []
    for bench_name, circuit, stats in results:
        ct_circuit = clifford_t_decompose(circuit)
        metrics = count_gates(ct_circuit)

        qasm_path = f"{args.qasm_dir}/{bench_name}.qasm"
        qasm_str = circuit_to_qasm(circuit, stats.get('num_qubits', 5))
        with open(qasm_path, 'w') as f:
            f.write(qasm_str)

        assert validate_qasm(qasm_str), f"Invalid QASM for {bench_name}"

        csv_results.append({
            'benchmark': bench_name,
            'cnot_count': metrics['cnot_count'],
            't_count': metrics['t_count'],
            'total_gates': metrics['total_gates'],
            'time': stats.get('time', 0),
        })

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['benchmark', 'cnot_count', 't_count', 'total_gates', 'time'])
        writer.writeheader()
        writer.writerows(csv_results)

    validate_file_exists(args.output, "Output CSV")

    total_cnot = sum(r['cnot_count'] for r in csv_results)
    total_t = sum(r['t_count'] for r in csv_results)
    total_gates = sum(r['total_gates'] for r in csv_results)
    print(f"GNN inference complete: {len(csv_results)} benchmarks, CNOT={total_cnot}, T={total_t}, gates={total_gates}")

    sys.exit(0)


if __name__ == '__main__':
    main()
