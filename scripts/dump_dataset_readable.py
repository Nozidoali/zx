import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ensure_dir
from src.debug import save_step_debug


def dump_supervised_readable(npz_path: str, output_dir: str, max_samples: int = 50):
    ensure_dir(output_dir)

    data = np.load(npz_path, allow_pickle=True)

    states = data['states']
    candidates = data['candidates']
    labels = data['labels']

    print(f"Dataset: {npz_path}")
    print(f"Total samples: {len(states):,}")
    print(f"Dumping first {min(max_samples, len(states))} to {output_dir}/")

    for i in range(min(max_samples, len(states))):
        state = states[i]
        cands = candidates[i]
        label = int(labels[i])

        chosen_action = cands[label] if label < len(cands) else None

        base_path = f"{output_dir}/sample_{i:06d}"

        save_step_debug(
            step_num=i,
            state=state,
            action=chosen_action,
            candidates=list(cands),
            label_idx=label,
            base_path=base_path
        )

    print(f"✅ Dumped {min(max_samples, len(states))} samples")
    print(f"   Location: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Dump supervised dataset in human-readable format')
    parser.add_argument('--train', type=str, default='artifacts/data/sup_train.npz')
    parser.add_argument('--val', type=str, default='artifacts/data/sup_val.npz')
    parser.add_argument('--output', type=str, default='artifacts/dataset_debug')
    parser.add_argument('--max-samples', type=int, default=50, help='Max samples per split')
    args = parser.parse_args()

    if Path(args.train).exists():
        train_dir = f"{args.output}/train"
        print("\n=== TRAINING SET ===")
        dump_supervised_readable(args.train, train_dir, args.max_samples)

    if Path(args.val).exists():
        val_dir = f"{args.output}/val"
        print("\n=== VALIDATION SET ===")
        dump_supervised_readable(args.val, val_dir, args.max_samples)

    print(f"\n✅ Dataset dump complete: {args.output}/")

    sys.exit(0)


if __name__ == '__main__':
    main()
