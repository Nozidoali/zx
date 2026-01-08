from typing import Dict, List, Tuple

import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.gnn import GNNPolicy
from src.diag import DiagramState, from_pyzx_graph
from src.act import generate_candidates, apply_action
from src.met import count_gates, clifford_t_decompose
from src.utils import log_verbose


def train_rl(model: GNNPolicy,
             benchmarks: List[Tuple[str, any]],
             config: Dict,
             verbose: bool = False,
             tensorboard_dir: str = None) -> Dict[str, List[float]]:
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.0001))

    writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None

    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'moving_avg_reward': [],
    }

    baseline_reward = 0.0
    baseline_alpha = 0.1

    for episode in range(config.get('episodes', 1000)):
        bench_name, graph = benchmarks[np.random.randint(len(benchmarks))]

        trajectory, total_reward, episode_length = run_episode(
            model, graph, config, verbose
        )

        baseline_reward = baseline_alpha * total_reward + (1 - baseline_alpha) * baseline_reward

        optimizer.zero_grad()
        loss = compute_policy_loss(trajectory, total_reward, baseline_reward, config)

        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        history['episode_rewards'].append(total_reward)
        history['episode_lengths'].append(episode_length)
        history['moving_avg_reward'].append(baseline_reward)

        if writer:
            writer.add_scalar('Reward/episode', total_reward, episode)
            writer.add_scalar('Reward/moving_avg', baseline_reward, episode)
            writer.add_scalar('Episode/length', episode_length, episode)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(history['episode_rewards'][-10:])
            avg_length = np.mean(history['episode_lengths'][-10:])
            log_verbose(f"Episode {episode+1}/{config.get('episodes', 1000)}: "
                       f"avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}, "
                       f"baseline={baseline_reward:.2f}", verbose)

    if writer:
        writer.close()

    return history


def run_episode(model: GNNPolicy,
                graph: any,
                config: Dict,
                verbose: bool = False) -> Tuple[List[Dict], float, int]:
    state = from_pyzx_graph(graph)
    trajectory = []
    max_steps = config.get('max_steps', 1000)

    for step in range(max_steps):
        if state.is_terminated():
            break

        candidates = generate_candidates(state)
        if not candidates:
            break

        try:
            action, log_prob, logits = model.sample_action(state, candidates)
        except Exception as e:
            log_verbose(f"Action sampling failed: {e}", verbose)
            break

        shaped_reward = 0.0
        if config.get('reward_shaping', False):
            edges_before = len(list(state.graph.edges()))

        try:
            apply_action(state, action)
        except Exception as e:
            log_verbose(f"Action application failed: {e}", verbose)
            trajectory.append({
                'log_prob': log_prob,
                'reward': -10.0,
                'entropy': 0.0,
            })
            break

        if config.get('reward_shaping', False):
            edges_after = len(list(state.graph.edges()))
            shaped_reward = -(edges_after - edges_before)

        entropy = -(torch.softmax(logits, dim=0) * torch.log_softmax(logits, dim=0)).sum().item()
        trajectory.append({
            'log_prob': log_prob,
            'reward': shaped_reward,
            'entropy': entropy,
        })

    terminal_reward = compute_terminal_reward(state, config)

    if trajectory:
        trajectory[-1]['reward'] += terminal_reward

    total_reward = sum(step['reward'] for step in trajectory)
    episode_length = len(trajectory)

    return trajectory, total_reward, episode_length


def compute_terminal_reward(state: DiagramState, config: Dict) -> float:
    circuit = clifford_t_decompose(state.circuit)

    counts = count_gates(circuit)

    alpha = config.get('alpha', 1.0)  # CNOT weight
    beta = config.get('beta', 10.0)   # T weight (expensive)
    gamma = config.get('gamma_gate', 0.1)  # Total gate weight

    weighted_count = (alpha * counts['cnot_count'] +
                     beta * counts['t_count'] +
                     gamma * counts['total_gates'])

    return -weighted_count


def compute_policy_loss(trajectory: List[Dict],
                       total_reward: float,
                       baseline: float,
                       config: Dict) -> torch.Tensor:
    if not trajectory:
        return None

    advantage = total_reward - baseline

    policy_loss = 0.0
    entropy_bonus = 0.0

    for step in trajectory:
        policy_loss -= step['log_prob'] * advantage
        entropy_bonus += step['entropy']

    entropy_coef = config.get('entropy_coef', 0.01)
    loss = policy_loss - entropy_coef * entropy_bonus

    return torch.tensor(loss, requires_grad=True)
