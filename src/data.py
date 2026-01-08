import json
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.diag import DiagramState
from src.act import Action


class SupervisedDataset(Dataset):
    """Dataset for supervised learning from expert traces."""

    def __init__(self, data_path: str):
        self.data = np.load(data_path, allow_pickle=True)
        self.states = self.data['states']
        self.candidates = self.data['candidates']
        self.labels = self.data['labels']

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[DiagramState, List[Action], int]:
        state = self.states[idx]
        candidates = self.candidates[idx]
        label = int(self.labels[idx])

        return state, candidates, label


class RLReplayBuffer:
    """Replay buffer for RL training."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.trajectories = []

    def add_trajectory(self, trajectory: List[Dict]) -> None:
        self.trajectories.append(trajectory)

        if len(self.trajectories) > self.capacity:
            self.trajectories.pop(0)

    def sample_batch(self, batch_size: int) -> List[List[Dict]]:
        indices = np.random.choice(len(self.trajectories),
                                   size=min(batch_size, len(self.trajectories)),
                                   replace=False)
        return [self.trajectories[i] for i in indices]

    def get_all(self) -> List[List[Dict]]:
        """Get all trajectories."""
        return self.trajectories

    def clear(self) -> None:
        """Clear buffer."""
        self.trajectories = []

    def __len__(self) -> int:
        return len(self.trajectories)


def collate_fn(batch: List[Tuple[DiagramState, List[Action], int]]) -> Tuple[List[DiagramState], List[List[Action]], torch.Tensor]:
    states = [item[0] for item in batch]
    candidates = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    return states, candidates, labels


def save_supervised_data(states: List[DiagramState],
                         candidates: List[List[Action]],
                         labels: List[int],
                         output_path: str) -> None:
    np.savez(output_path,
             states=np.array(states, dtype=object),
             candidates=np.array(candidates, dtype=object),
             labels=np.array(labels, dtype=int))


def load_traces_from_jsonl(trace_file: str) -> List[Dict]:
    traces = []
    with open(trace_file, 'r') as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces
