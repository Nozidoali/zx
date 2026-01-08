import argparse
import sys
from pathlib import Path
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ensure_dir
from src.act import CNOTAction, PhaseAction, CliffordAction


def action_to_dict(action):
    if isinstance(action, CNOTAction):
        return {'type': 'CNode', 'u': action.u, 'v': action.v}
    elif isinstance(action, PhaseAction):
        return {'type': 'Face', 'vertices': list(action.vertices)}
    elif isinstance(action, CliffordAction):
        return {'type': 'Clifford', 'qubit': action.qubit, 'gate': action.gate}
    else:
        return {'type': 'Unknown', 'repr': str(action)}


def dump_readable_dataset(npz_path: str, output_dir: str, max_samples: int = 100):
    ensure_dir(output_dir)

    data = np.load(npz_path, allow_pickle=True)

    states = data['states']
    candidates = data['candidates']
    labels = data['labels']

    summary = {
        'total_samples': len(states),
        'data_keys': list(data.keys()),
        'output_dir': output_dir,
        'samples_dumped': min(max_samples, len(states))
    }

    print(f"Dataset: {npz_path}")
    print(f"Total samples: {len(states):,}")
    print(f"Dumping first {min(max_samples, len(states))} to {output_dir}/")

    for i in range(min(max_samples, len(states))):
        state = states[i]

        state_summary = {
            'num_qubits': state.num_qubits,
            'frontier': list(state.frontier),
            'num_nodes': len(list(state.graph.vertices())),
            'num_edges': len(list(state.graph.edges())),
            'circuit_length': len(state.circuit)
        }

        cands = candidates[i]
        cands_list = [action_to_dict(c) for c in cands]

        sample = {
            'sample_id': i,
            'state': state_summary,
            'num_candidates': len(cands),
            'candidates': cands_list,
            'label': int(labels[i]),
            'chosen_action': cands_list[int(labels[i])] if int(labels[i]) < len(cands_list) else None
        }

        output_file = f"{output_dir}/sample_{i:06d}.json"
        with open(output_file, 'w') as f:
            json.dump(sample, f, indent=2)

    summary_file = f"{output_dir}/summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Dumped {summary['samples_dumped']} samples")
    print(f"   Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Inspect and dump dataset in readable format')
    parser.add_argument('--train', type=str, default='artifacts/data/sup_train.npz')
    parser.add_argument('--val', type=str, default='artifacts/data/sup_val.npz')
    parser.add_argument('--output', type=str, default='artifacts/dataset_readable')
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples to dump per dataset')
    args = parser.parse_args()

    if Path(args.train).exists():
        train_dir = f"{args.output}/train"
        dump_readable_dataset(args.train, train_dir, args.max_samples)

    if Path(args.val).exists():
        val_dir = f"{args.output}/val"
        dump_readable_dataset(args.val, val_dir, args.max_samples)

    print(f"\n✅ Dataset inspection complete!")
    print(f"   Output: {args.output}/")

    sys.exit(0)


if __name__ == '__main__':
    main()
