import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn import GNNPolicy
from src.data import SupervisedDataset
from src.sup import train_supervised
from src.utils import set_seed, ensure_dir, validate_file_exists


def main():
    parser = argparse.ArgumentParser(description='Supervised training')
    parser.add_argument('--data', type=str, default='artifacts/data/sup_train.npz')
    parser.add_argument('--output', type=str, default='artifacts/models/sup.pt')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--kind-dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    validate_file_exists(args.data, "Training data")
    ensure_dir(str(Path(args.output).parent))

    tensorboard_log_dir = 'artifacts/runs/sup' if args.tensorboard else None
    if tensorboard_log_dir:
        ensure_dir(tensorboard_log_dir)

    dataset = SupervisedDataset(args.data)

    model = GNNPolicy(
        kind_dim=args.kind_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    config = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'val_split': 0.2,
        'patience': args.patience,
    }

    print(f"Starting supervised training with {args.epochs} epochs...")
    print(f"Dataset: {len(dataset)} samples")
    if tensorboard_log_dir:
        print(f"TensorBoard logs: {tensorboard_log_dir}")
        print(f"Run: tensorboard --logdir={tensorboard_log_dir}")

    history = train_supervised(model, dataset, config, args.verbose, tensorboard_log_dir)

    torch.save(model.state_dict(), args.output)
    validate_file_exists(args.output, "Model checkpoint")

    final_train_loss = history['train_loss'][-1] if history['train_loss'] else float('inf')
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
    final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0.0
    final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0.0

    assert final_val_loss < 10.0, f"Val loss too high: {final_val_loss}"

    print(f"\nTraining complete!")
    print(f"Train: loss={final_train_loss:.4f}, acc={final_train_acc:.4f}")
    print(f"Val:   loss={final_val_loss:.4f}, acc={final_val_acc:.4f}")
    print(f"Model saved to: {args.output}")

    sys.exit(0)


if __name__ == '__main__':
    main()
