import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn import GNNPolicy
from src.bench import get_benchmark_list
from src.rl import train_rl
from src.utils import set_seed, ensure_dir, validate_file_exists


def main():
    parser = argparse.ArgumentParser(description='RL training')
    parser.add_argument('--init', type=str, required=True, help='Initial checkpoint')
    parser.add_argument('--output', type=str, default='artifacts/models/rl.pt')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--kind-dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--reward-shaping', action='store_true')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--gamma-gate', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    validate_file_exists(args.init, "Init checkpoint")
    ensure_dir(str(Path(args.output).parent))

    tensorboard_log_dir = 'artifacts/runs/rl' if args.tensorboard else None
    if tensorboard_log_dir:
        ensure_dir(tensorboard_log_dir)

    model = GNNPolicy(
        kind_dim=args.kind_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    model.load_state_dict(torch.load(args.init))

    benchmarks = get_benchmark_list()

    config = {
        'episodes': args.episodes,
        'gamma': args.gamma,
        'lr': args.lr,
        'entropy_coef': args.entropy_coef,
        'reward_shaping': args.reward_shaping,
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma_gate': args.gamma_gate,
        'max_steps': 1000,
    }

    print(f"Starting RL training with {args.episodes} episodes...")
    if tensorboard_log_dir:
        print(f"TensorBoard logs: {tensorboard_log_dir}")
        print(f"Run: tensorboard --logdir={tensorboard_log_dir}")

    history = train_rl(model, benchmarks, config, args.verbose, tensorboard_log_dir)

    torch.save(model.state_dict(), args.output)
    validate_file_exists(args.output, "RL checkpoint")

    import numpy as np
    final_avg = np.mean(history['episode_rewards'][-100:])

    print(f"\nRL training complete!")
    print(f"Final avg reward (last 100): {final_avg:.2f}")
    print(f"Model saved to: {args.output}")

    sys.exit(0)


if __name__ == '__main__':
    main()
