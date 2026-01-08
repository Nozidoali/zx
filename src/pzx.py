import time
from typing import Dict, Tuple, List

import pyzx as zx

from src.met import count_gates, clifford_t_decompose
from src.diag import DiagramState, from_pyzx_graph
from src.act import generate_candidates, apply_action
from src.exp import dump_trace_step
from src.utils import log_verbose


def run_pyzx_baseline(graph: zx.Graph, verbose: bool = False) -> Tuple[List, Dict]:
    start_time = time.time()

    try:
        g = graph.copy()
        circuit = zx.extract_circuit(g)

        our_circuit = []
        if hasattr(circuit, 'gates'):
            for gate in circuit.gates:
                gate_name = gate.name if hasattr(gate, 'name') else str(type(gate).__name__)
                qubits = [gate.control, gate.target] if hasattr(gate, 'control') else [gate.target]
                params = [gate.phase] if hasattr(gate, 'phase') else []
                our_circuit.append((gate_name, qubits, params))

        ct_circuit = clifford_t_decompose(our_circuit)

        metrics = count_gates(ct_circuit)
        metrics['time'] = time.time() - start_time

        log_verbose(f"PyZX baseline: {metrics}", verbose)

        return ct_circuit, metrics

    except Exception as e:
        log_verbose(f"PyZX extraction failed: {e}", verbose)
        return [], {
            'cnot_count': 0,
            't_count': 0,
            'total_gates': 0,
            'time': time.time() - start_time,
            'error': str(e),
        }


def run_pyzx_with_trace(graph: zx.Graph, trace_file: str, verbose: bool = False) -> Tuple[List, Dict]:

    state = from_pyzx_graph(graph)
    max_steps = 1000

    for step in range(max_steps):
        if state.is_terminated():
            break

        candidates = generate_candidates(state)
        if not candidates:
            break

        from src.act import PhaseAction, CNOTAction, CliffordAction

        phase_actions = [a for a in candidates if isinstance(a, PhaseAction)]
        cnot_actions = [a for a in candidates if isinstance(a, CNOTAction)]
        clifford_actions = [a for a in candidates if isinstance(a, CliffordAction)]

        if phase_actions:
            action = phase_actions[0]
        elif cnot_actions:
            action = min(cnot_actions, key=lambda a: (a.u, a.v))
        elif clifford_actions:
            action = clifford_actions[0]
        else:
            break

        dump_trace_step(state, action, step, trace_file)

        try:
            apply_action(state, action)
        except Exception as e:
            log_verbose(f"Action failed at step {step}: {e}", verbose)
            break

    ct_circuit = clifford_t_decompose(state.circuit)
    metrics = count_gates(ct_circuit)

    return ct_circuit, metrics
