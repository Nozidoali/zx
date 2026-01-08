"""Dataset and replay buffer for supervised and RL training."""
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
        """Load supervised dataset from .npz file.
        
        Args:
            data_path: Path to .npz file
        """
        self.data = np.load(data_path, allow_pickle=True)
        self.states = self.data['states']
        self.candidates = self.data['candidates']
        self.labels = self.data['labels']
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[DiagramState, List[Action], int]:
        """Get a single training example.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (state, candidates, label_idx)
        """
        state = self.states[idx]
        candidates = self.candidates[idx]
        label = int(self.labels[idx])
        
        return state, candidates, label


class RLReplayBuffer:
    """Replay buffer for RL training."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of trajectories to store
        """
        self.capacity = capacity
        self.trajectories = []
    
    def add_trajectory(self, trajectory: List[Dict]) -> None:
        """Add a trajectory to the buffer.
        
        Args:
            trajectory: List of step dictionaries with keys:
                - state: DiagramState
                - action: Action
                - log_prob: float
                - reward: float
        """
        self.trajectories.append(trajectory)
        
        # Remove oldest if over capacity
        if len(self.trajectories) > self.capacity:
            self.trajectories.pop(0)
    
    def sample_batch(self, batch_size: int) -> List[List[Dict]]:
        """Sample random batch of trajectories.
        
        Args:
            batch_size: Number of trajectories to sample
            
        Returns:
            List of trajectories
        """
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
    """Collate function for supervised dataset batching.
    
    Args:
        batch: List of (state, candidates, label) tuples
        
    Returns:
        Tuple of (states, candidates, labels)
    """
    states = [item[0] for item in batch]
    candidates = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
    return states, candidates, labels


def save_supervised_data(states: List[DiagramState], 
                         candidates: List[List[Action]], 
                         labels: List[int],
                         output_path: str) -> None:
    """Save supervised dataset to .npz file.
    
    Args:
        states: List of diagram states
        candidates: List of candidate lists
        labels: List of label indices
        output_path: Output file path
    """
    np.savez(output_path,
             states=np.array(states, dtype=object),
             candidates=np.array(candidates, dtype=object),
             labels=np.array(labels, dtype=int))


def load_traces_from_jsonl(trace_file: str) -> List[Dict]:
    """Load trace steps from JSONL file.
    
    Args:
        trace_file: Path to trace file
        
    Returns:
        List of trace step dictionaries
    """
    traces = []
    with open(trace_file, 'r') as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces

