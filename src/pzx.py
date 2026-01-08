"""PyZX baseline extraction wrapper with tracing support."""
import time
from typing import Dict, Tuple, List

import pyzx as zx

from src.met import count_gates, clifford_t_decompose
from src.diag import DiagramState, from_pyzx_graph
from src.act import generate_candidates, apply_action
from src.exp import dump_trace_step
from src.utils import log_verbose


def run_pyzx_baseline(graph: zx.Graph, verbose: bool = False) -> Tuple[List, Dict]:
    """Run PyZX baseline extraction.
    
    Args:
        graph: PyZX graph
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (circuit, metrics)
    """
    start_time = time.time()
    
    try:
        # Use PyZX's built-in extraction
        g = graph.copy()
        circuit = zx.extract_circuit(g)
        
        # Convert to our circuit format
        our_circuit = []
        if hasattr(circuit, 'gates'):
            for gate in circuit.gates:
                gate_name = gate.name if hasattr(gate, 'name') else str(type(gate).__name__)
                qubits = [gate.control, gate.target] if hasattr(gate, 'control') else [gate.target]
                params = [gate.phase] if hasattr(gate, 'phase') else []
                our_circuit.append((gate_name, qubits, params))
        
        # Clifford+T decomposition
        ct_circuit = clifford_t_decompose(our_circuit)
        
        # Count gates
        metrics = count_gates(ct_circuit)
        metrics['time'] = time.time() - start_time
        
        log_verbose(f"PyZX baseline: {metrics}", verbose)
        
        return ct_circuit, metrics
        
    except Exception as e:
        log_verbose(f"PyZX extraction failed: {e}", verbose)
        # Return empty circuit with error
        return [], {
            'cnot_count': 0,
            't_count': 0,
            'total_gates': 0,
            'time': time.time() - start_time,
            'error': str(e),
        }


def run_pyzx_with_trace(graph: zx.Graph, trace_file: str, verbose: bool = False) -> Tuple[List, Dict]:
    """Run PyZX extraction with tracing for supervised learning.
    
    This is a simplified version that generates synthetic traces based on
    a heuristic policy, since PyZX doesn't expose internal extraction steps.
    
    Args:
        graph: PyZX graph
        trace_file: Path to output trace file
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (circuit, metrics)
    """
    # For supervised learning, we'll use a simple heuristic policy
    # Real implementation would hook into PyZX internals or use a different baseline
    
    state = from_pyzx_graph(graph)
    max_steps = 1000
    
    for step in range(max_steps):
        if state.is_terminated():
            break
        
        candidates = generate_candidates(state)
        if not candidates:
            break
        
        # Heuristic: prefer Face actions, then CNode, then Clifford
        # This creates training data for supervised learning
        from src.act import FaceAction, CNodeAction, CliffordAction
        
        face_actions = [a for a in candidates if isinstance(a, FaceAction)]
        cnode_actions = [a for a in candidates if isinstance(a, CNodeAction)]
        clifford_actions = [a for a in candidates if isinstance(a, CliffordAction)]
        
        if face_actions:
            action = face_actions[0]
        elif cnode_actions:
            # Pick pair with smallest node IDs (deterministic heuristic)
            action = min(cnode_actions, key=lambda a: (a.u, a.v))
        elif clifford_actions:
            action = clifford_actions[0]
        else:
            break
        
        # Record trace step
        dump_trace_step(state, action, step, trace_file)
        
        # Apply action
        try:
            apply_action(state, action)
        except Exception as e:
            log_verbose(f"Action failed at step {step}: {e}", verbose)
            break
    
    # Get metrics
    ct_circuit = clifford_t_decompose(state.circuit)
    metrics = count_gates(ct_circuit)
    
    return ct_circuit, metrics

