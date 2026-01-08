import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyzx as zx
from src.diag import DiagramState, from_pyzx_graph
from src.act import CNodeAction, FaceAction, CliffordAction, generate_candidates, apply_action
from src.qasm import circuit_to_qasm, validate_qasm
from src.gnn import GNNPolicy


def test_cnode_rejects_non_frontier():
    graph = zx.generate.cliffordT(3, 10)
    state = from_pyzx_graph(graph)
    
    non_frontier = [v for v in graph.vertices() if v not in state.frontier]
    if len(non_frontier) >= 2:
        action = CNodeAction(non_frontier[0], non_frontier[1])
        with pytest.raises(AssertionError):
            apply_action(state, action)


def test_face_requires_deg_1():
    graph = zx.generate.cliffordT(3, 10)
    state = from_pyzx_graph(graph)
    
    for v in state.frontier:
        if state.graph.vertex_degree(v) > 1:
            action = FaceAction(v, 0.5)
            with pytest.raises(AssertionError):
                apply_action(state, action)
            break


def test_circuit_gates_restricted():
    circuit = [('CZ', [0, 1], []), ('Rz', [0], [0.5]), ('H', [1], [])]
    qasm = circuit_to_qasm(circuit, 3)
    assert 'cz q[0],q[1]' in qasm
    assert 'rz(0.5) q[0]' in qasm
    assert 'h q[1]' in qasm
    
    invalid_circuit = [('INVALID_GATE', [0], [])]
    with pytest.raises(AssertionError):
        circuit_to_qasm(invalid_circuit, 3)


def test_qasm_validation():
    valid_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];
"""
    assert validate_qasm(valid_qasm)
    
    invalid_qasm = "not valid qasm"
    assert not validate_qasm(invalid_qasm)


def test_gnn_forward():
    model = GNNPolicy(kind_dim=8, hidden_dim=32, num_layers=2)
    graph = zx.generate.cliffordT(3, 10)
    state = from_pyzx_graph(graph)
    
    candidates = generate_candidates(state)
    if candidates:
        logits = model(state, candidates)
        assert logits.shape[0] == len(candidates)

