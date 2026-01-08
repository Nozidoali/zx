"""Action definitions and application logic for ZX diagram extraction."""
from dataclasses import dataclass
from typing import List, Union

import pyzx as zx
from src.diag import DiagramState, EPS


@dataclass
class CNodeAction:
    """Add CNode between two frontier nodes."""
    u: int
    v: int
    
    def __str__(self):
        return f"CNode({self.u}, {self.v})"


@dataclass
class FaceAction:
    """Extract face gadget (Z-rotation) from frontier node."""
    u: int
    theta: float
    
    def __str__(self):
        return f"Rz({self.u}, {self.theta:.4f})"


@dataclass
class CliffordAction:
    """Apply single-qubit Clifford gate."""
    u: int
    kind: str  # 'H', 'S', 'X', 'Z', etc.
    
    def __str__(self):
        return f"{self.kind}({self.u})"


Action = Union[CNodeAction, FaceAction, CliffordAction]


def validate_action(state: DiagramState, action: Action) -> None:
    """Validate action preconditions (with assertions).
    
    Args:
        state: Current diagram state
        action: Action to validate
    """
    if isinstance(action, CNodeAction):
        assert action.u in state.frontier, f"Node {action.u} not in frontier"
        assert action.v in state.frontier, f"Node {action.v} not in frontier"
        assert action.u != action.v, f"Cannot create CNode with same node"
        
    elif isinstance(action, FaceAction):
        assert action.u in state.frontier, f"Node {action.u} not in frontier"
        assert state.graph.vertex_degree(action.u) == 1, \
            f"Node {action.u} must have degree 1 for face extraction"
        assert state.graph.type(action.u) == zx.VertexType.Z, \
            f"Node {action.u} must be Z-spider for face extraction"
        phase = state.graph.phase(action.u)
        assert abs(phase) > EPS, f"Node {action.u} has trivial phase"
        
    elif isinstance(action, CliffordAction):
        assert action.u in state.frontier, f"Node {action.u} not in frontier"
        # Additional Clifford-specific predicates would go here


def apply_action(state: DiagramState, action: Action) -> None:
    """Apply action to diagram state (modifies in-place).
    
    Args:
        state: Diagram state to modify
        action: Action to apply
    """
    validate_action(state, action)
    
    if isinstance(action, CNodeAction):
        _apply_cnode(state, action)
    elif isinstance(action, FaceAction):
        _apply_face(state, action)
    elif isinstance(action, CliffordAction):
        _apply_clifford(state, action)


def _apply_cnode(state: DiagramState, action: CNodeAction) -> None:
    """Apply CNode action: emit gate, apply bi-algebra, update frontier.
    
    Args:
        state: Diagram state
        action: CNode action
    """
    u, v = action.u, action.v
    
    # Emit CNode to circuit (map nodes to qubits - simplified)
    qubit_u = _node_to_qubit(state, u)
    qubit_v = _node_to_qubit(state, v)
    state.add_to_circuit('CZ', [qubit_u, qubit_v])
    
    # Apply bi-algebra rewrite: add edge between u and v in diagram
    # This is a simplified version - full bi-algebra would involve graph rewriting
    if not state.graph.connected(u, v):
        state.graph.add_edge((u, v))
    
    # Update frontier: remove u, v and add their non-frontier neighbors
    state.frontier.discard(u)
    state.frontier.discard(v)
    
    for node in [u, v]:
        for neighbor in state.graph.neighbors(node):
            if neighbor not in state.frontier and \
               state.graph.type(neighbor) != zx.VertexType.BOUNDARY:
                state.frontier.add(neighbor)


def _apply_face(state: DiagramState, action: FaceAction) -> None:
    """Apply face gadget extraction: emit Rz, remove phase, update frontier.
    
    Args:
        state: Diagram state
        action: Face action
    """
    u = action.u
    theta = action.theta
    
    # Emit Rz gate
    qubit = _node_to_qubit(state, u)
    state.add_to_circuit('Rz', [qubit], [theta])
    
    # Remove phase from node
    state.graph.set_phase(u, 0)
    
    # Update frontier: remove u and add its neighbors
    state.frontier.discard(u)
    neighbor = list(state.graph.neighbors(u))[0]  # degree == 1
    if state.graph.type(neighbor) != zx.VertexType.BOUNDARY:
        state.frontier.add(neighbor)


def _apply_clifford(state: DiagramState, action: CliffordAction) -> None:
    """Apply Clifford gate: emit gate, update diagram, update frontier.
    
    Args:
        state: Diagram state
        action: Clifford action
    """
    u = action.u
    kind = action.kind
    
    # Emit Clifford gate
    qubit = _node_to_qubit(state, u)
    state.add_to_circuit(kind, [qubit])
    
    # Update diagram based on Clifford type (simplified)
    if kind == 'H':
        # Hadamard swaps Z and X basis
        if state.graph.type(u) == zx.VertexType.Z:
            state.graph.set_type(u, zx.VertexType.X)
        elif state.graph.type(u) == zx.VertexType.X:
            state.graph.set_type(u, zx.VertexType.Z)
    
    # Frontier update (simplified - may need more sophisticated logic)
    # For now, keep node in frontier


def _node_to_qubit(state: DiagramState, node: int) -> int:
    """Map node ID to qubit index (simplified).
    
    Args:
        state: Diagram state
        node: Node ID
        
    Returns:
        Qubit index (0-based)
    """
    # This is a simplified mapping - would need proper tracking in real implementation
    return hash(node) % state.num_qubits


def generate_candidates(state: DiagramState, max_pairs: int = 100) -> List[Action]:
    """Generate list of valid candidate actions.
    
    Args:
        state: Current diagram state
        max_pairs: Maximum number of CNode pairs to consider
        
    Returns:
        List of valid actions
    """
    candidates = []
    frontier_list = list(state.frontier)
    
    # CNode candidates: all frontier pairs (or top-k)
    cnode_pairs = []
    for i, u in enumerate(frontier_list):
        for v in frontier_list[i+1:]:
            cnode_pairs.append(CNodeAction(u, v))
    
    # Limit pairs if too many
    if len(cnode_pairs) > max_pairs:
        # Simple heuristic: keep first max_pairs
        cnode_pairs = cnode_pairs[:max_pairs]
    
    candidates.extend(cnode_pairs)
    
    # Face candidates: frontier nodes with deg==1 and non-trivial phase
    for u in frontier_list:
        if state.graph.vertex_degree(u) == 1 and \
           state.graph.type(u) == zx.VertexType.Z:
            phase = state.graph.phase(u)
            if abs(phase) > EPS:
                candidates.append(FaceAction(u, phase))
    
    # Clifford candidates: eligible frontier nodes
    for u in frontier_list:
        # Simple predicate: allow Hadamard on Z or X spiders
        vtype = state.graph.type(u)
        if vtype in [zx.VertexType.Z, zx.VertexType.X]:
            candidates.append(CliffordAction(u, 'H'))
            
            # Also allow S gate on Z spiders
            if vtype == zx.VertexType.Z:
                candidates.append(CliffordAction(u, 'S'))
    
    return candidates

