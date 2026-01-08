from dataclasses import dataclass
from typing import List, Union

import pyzx as zx
from src.diag import DiagramState, EPS


@dataclass
class CNOTAction:
    u: int
    v: int

    def __str__(self):
        return f"CNOT({self.u}, {self.v})"


@dataclass
class PhaseAction:
    u: int
    theta: float

    def __str__(self):
        return f"Rz({self.u}, {self.theta:.4f})"


@dataclass
class CliffordAction:
    u: int
    kind: str

    def __str__(self):
        return f"{self.kind}({self.u})"


Action = Union[CNOTAction, PhaseAction, CliffordAction]


def validate_action(state: DiagramState, action: Action) -> None:
    if isinstance(action, CNOTAction):
        assert action.u in state.frontier, f"Node {action.u} not in frontier"
        assert action.v in state.frontier, f"Node {action.v} not in frontier"
        assert action.u != action.v, f"Cannot create CNOT with same node"

    elif isinstance(action, PhaseAction):
        assert action.u in state.frontier, f"Node {action.u} not in frontier"
        assert state.graph.vertex_degree(action.u) == 1, \
            f"Node {action.u} must have degree 1 for phase extraction"
        assert state.graph.type(action.u) == zx.VertexType.Z, \
            f"Node {action.u} must be Z-spider for face extraction"
        phase = state.graph.phase(action.u)
        assert abs(phase) > EPS, f"Node {action.u} has trivial phase"

    elif isinstance(action, CliffordAction):
        assert action.u in state.frontier, f"Node {action.u} not in frontier"


def apply_action(state: DiagramState, action: Action) -> None:
    validate_action(state, action)

    if isinstance(action, CNOTAction):
        _apply_cnot(state, action)
    elif isinstance(action, PhaseAction):
        _apply_phase(state, action)
    elif isinstance(action, CliffordAction):
        _apply_clifford(state, action)


def _apply_cnot(state: DiagramState, action: CNOTAction) -> None:
    u, v = action.u, action.v

    qubit_u = _node_to_qubit(state, u)
    qubit_v = _node_to_qubit(state, v)
    state.add_to_circuit('CZ', [qubit_u, qubit_v])

    if not state.graph.connected(u, v):
        state.graph.add_edge((u, v))

    state.frontier.discard(u)
    state.frontier.discard(v)

    for node in [u, v]:
        for neighbor in state.graph.neighbors(node):
            if neighbor not in state.frontier and \
               state.graph.type(neighbor) != zx.VertexType.BOUNDARY:
                state.frontier.add(neighbor)


def _apply_phase(state: DiagramState, action: PhaseAction) -> None:
    u = action.u
    theta = action.theta

    qubit = _node_to_qubit(state, u)
    state.add_to_circuit('Rz', [qubit], [theta])

    state.graph.set_phase(u, 0)

    state.frontier.discard(u)
    neighbor = list(state.graph.neighbors(u))[0]
    if state.graph.type(neighbor) != zx.VertexType.BOUNDARY:
        state.frontier.add(neighbor)


def _apply_clifford(state: DiagramState, action: CliffordAction) -> None:
    u = action.u
    kind = action.kind

    qubit = _node_to_qubit(state, u)
    state.add_to_circuit(kind, [qubit])

    if kind == 'H':
        if state.graph.type(u) == zx.VertexType.Z:
            state.graph.set_type(u, zx.VertexType.X)
        elif state.graph.type(u) == zx.VertexType.X:
            state.graph.set_type(u, zx.VertexType.Z)



def _node_to_qubit(state: DiagramState, node: int) -> int:
    return hash(node) % state.num_qubits


def generate_candidates(state: DiagramState, max_pairs: int = 100) -> List[Action]:
    candidates = []
    frontier_list = list(state.frontier)

    cnot_pairs = []
    for i, u in enumerate(frontier_list):
        for v in frontier_list[i+1:]:
            cnot_pairs.append(CNOTAction(u, v))

    if len(cnot_pairs) > max_pairs:
        cnot_pairs = cnot_pairs[:max_pairs]

    candidates.extend(cnot_pairs)

    for u in frontier_list:
        if state.graph.vertex_degree(u) == 1 and \
           state.graph.type(u) == zx.VertexType.Z:
            phase = state.graph.phase(u)
            if abs(phase) > EPS:
                candidates.append(PhaseAction(u, phase))

    for u in frontier_list:
        vtype = state.graph.type(u)
        if vtype in [zx.VertexType.Z, zx.VertexType.X]:
            candidates.append(CliffordAction(u, 'H'))

            if vtype == zx.VertexType.Z:
                candidates.append(CliffordAction(u, 'S'))

    return candidates
