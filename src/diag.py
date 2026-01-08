"""Diagram state representation and node embeddings."""
import math
from typing import Dict, List, Set, Tuple

import numpy as np
import pyzx as zx
import torch


# Node type constants
NODE_Z = 0
NODE_X = 1
NODE_H = 2
NODE_BOUNDARY = 3
NODE_OTHER = 4
NUM_NODE_TYPES = 5

EPS = 1e-6


class DiagramState:
    """Wrapper for ZX diagram with frontier tracking and circuit extraction."""
    
    def __init__(self, graph: zx.Graph):
        """Initialize diagram state from PyZX graph.
        
        Args:
            graph: PyZX graph object
        """
        self.graph = graph.copy()
        self.frontier = self._init_frontier()
        self.circuit = []  # List of (gate_type, qubits, params)
        inputs = list(self.graph.inputs()) if hasattr(self.graph, 'inputs') else []
        self.num_qubits = len([v for v in inputs if self.graph.type(v) == zx.VertexType.BOUNDARY])
        if self.num_qubits == 0:
            self.num_qubits = max(3, len([v for v in self.graph.vertices() if self.graph.type(v) == zx.VertexType.BOUNDARY]) // 2)
    
    def _init_frontier(self) -> Set[int]:
        """Initialize frontier as input boundary nodes."""
        frontier = set()
        for v in self.graph.vertices():
            if self.graph.type(v) == zx.VertexType.BOUNDARY:
                # Add neighbors of input boundaries
                for neighbor in self.graph.neighbors(v):
                    frontier.add(neighbor)
        if not frontier:
            frontier = set(list(self.graph.vertices())[:5])
        return frontier
    
    def get_frontier(self) -> Set[int]:
        """Return current frontier set."""
        return self.frontier.copy()
    
    def is_terminated(self) -> bool:
        """Check if extraction is complete (frontier reached outputs)."""
        # Simple termination: frontier is empty or all remaining nodes are outputs
        if not self.frontier:
            return True
        
        # Check if we can't make progress
        remaining = len([v for v in self.graph.vertices() 
                        if self.graph.type(v) not in [zx.VertexType.BOUNDARY]])
        return remaining == 0
    
    def get_node_features(self, v: int) -> Dict[str, float]:
        """Extract features for a node.
        
        Args:
            v: Vertex ID
            
        Returns:
            Dictionary of node features
        """
        node_type = self._get_node_kind(v)
        phase = self.graph.phase(v)
        
        return {
            'kind': node_type,
            'sin_theta': math.sin(phase * math.pi) if abs(phase) > EPS else 0.0,
            'cos_theta': math.cos(phase * math.pi) if abs(phase) > EPS else 1.0,
            'is_frontier': float(v in self.frontier),
            'degree': float(self.graph.vertex_degree(v)),
            'is_boundary': float(self.graph.type(v) == zx.VertexType.BOUNDARY),
        }
    
    def _get_node_kind(self, v: int) -> int:
        """Get node type ID."""
        vtype = self.graph.type(v)
        
        if vtype == zx.VertexType.Z:
            return NODE_Z
        elif vtype == zx.VertexType.X:
            return NODE_X
        elif vtype == zx.VertexType.H_BOX:
            return NODE_H
        elif vtype == zx.VertexType.BOUNDARY:
            return NODE_BOUNDARY
        else:
            return NODE_OTHER
    
    def add_to_circuit(self, gate_type: str, qubits: List[int], params: List[float] = None):
        """Add gate to extracted circuit."""
        self.circuit.append((gate_type, qubits, params or []))
    
    def copy(self):
        """Create a deep copy of this state."""
        new_state = DiagramState(self.graph)
        new_state.frontier = self.frontier.copy()
        new_state.circuit = self.circuit.copy()
        return new_state


def node_kind_encoding(vertex_type: int) -> int:
    """Get node kind ID from vertex type.
    
    Args:
        vertex_type: PyZX vertex type
        
    Returns:
        Node kind integer ID
    """
    if vertex_type == zx.VertexType.Z:
        return NODE_Z
    elif vertex_type == zx.VertexType.X:
        return NODE_X
    elif vertex_type == zx.VertexType.H_BOX:
        return NODE_H
    elif vertex_type == zx.VertexType.BOUNDARY:
        return NODE_BOUNDARY
    else:
        return NODE_OTHER


def get_node_embedding(state: DiagramState, v: int, kind_embedding: torch.nn.Embedding) -> torch.Tensor:
    """Create full embedding for a node.
    
    Args:
        state: Diagram state
        v: Vertex ID
        kind_embedding: Embedding layer for node types
        
    Returns:
        Concatenated embedding tensor
    """
    features = state.get_node_features(v)
    
    # Get kind embedding
    kind_id = torch.tensor([features['kind']], dtype=torch.long)
    kind_emb = kind_embedding(kind_id).squeeze(0)
    
    # Concatenate with other features
    other_features = torch.tensor([
        features['sin_theta'],
        features['cos_theta'],
        features['is_frontier'],
        features['degree'],
        features['is_boundary'],
    ], dtype=torch.float32)
    
    return torch.cat([kind_emb, other_features])


def to_pyzx_graph(state: DiagramState) -> zx.Graph:
    """Convert diagram state to PyZX graph.
    
    Args:
        state: Diagram state
        
    Returns:
        PyZX graph object
    """
    return state.graph.copy()


def from_pyzx_graph(graph: zx.Graph) -> DiagramState:
    """Create diagram state from PyZX graph.
    
    Args:
        graph: PyZX graph
        
    Returns:
        New DiagramState object
    """
    return DiagramState(graph)

