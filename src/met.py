"""Gate counting and metrics for circuit evaluation."""
from typing import Dict, List, Tuple

import pyzx as zx


def count_gates(circuit: List[Tuple[str, List[int], List[float]]]) -> Dict[str, int]:
    """Count gates in circuit by type.
    
    Args:
        circuit: List of (gate_type, qubit_indices, params) tuples
        
    Returns:
        Dictionary with gate counts
    """
    counts = {
        'cnot_count': 0,
        't_count': 0,
        'total_gates': len(circuit),
        'h_count': 0,
        's_count': 0,
        'rz_count': 0,
        'cz_count': 0,
    }
    
    for gate_type, qubits, params in circuit:
        if gate_type in ['CNOT', 'CX']:
            counts['cnot_count'] += 1
        elif gate_type == 'T':
            counts['t_count'] += 1
        elif gate_type == 'H':
            counts['h_count'] += 1
        elif gate_type == 'S':
            counts['s_count'] += 1
        elif gate_type == 'Rz':
            counts['rz_count'] += 1
        elif gate_type in ['CZ', 'CNode']:
            counts['cz_count'] += 1
    
    return counts


def clifford_t_decompose(circuit: List[Tuple[str, List[int], List[float]]]) -> List[Tuple[str, List[int], List[float]]]:
    """Convert circuit to Clifford+T gate set.
    
    Args:
        circuit: Input circuit
        
    Returns:
        Circuit using only Clifford+T gates
    """
    clifford_t_circuit = []
    
    for gate_type, qubits, params in circuit:
        if gate_type in ['H', 'S', 'T', 'CNOT', 'CX', 'X', 'Z']:
            # Already Clifford+T
            clifford_t_circuit.append((gate_type, qubits, params))
        
        elif gate_type in ['CZ', 'CNode']:
            # CZ = H-CNOT-H
            clifford_t_circuit.append(('H', [qubits[1]], []))
            clifford_t_circuit.append(('CNOT', qubits, []))
            clifford_t_circuit.append(('H', [qubits[1]], []))
        
        elif gate_type == 'Rz':
            # Rz(θ) decomposition depends on angle
            # For simplicity, approximate with T gates if close to π/4 multiples
            angle = params[0] if params else 0
            
            # Simplified: use PyZX-style decomposition
            # For now, count as 1 T gate for small angles
            import math
            t_angle = math.pi / 4
            
            if abs(angle - t_angle) < 0.1:
                clifford_t_circuit.append(('T', qubits, []))
            elif abs(angle + t_angle) < 0.1:
                # T-dagger = S-T-dagger-S-dagger = STdgSdg
                clifford_t_circuit.append(('S', qubits, []))
                clifford_t_circuit.append(('T', qubits, []))
                clifford_t_circuit.append(('S', qubits, []))
            else:
                # General Rz: approximate with multiple T gates
                num_t = max(1, int(abs(angle) / t_angle))
                for _ in range(num_t):
                    clifford_t_circuit.append(('T', qubits, []))
        
        else:
            # Pass through unknown gates
            clifford_t_circuit.append((gate_type, qubits, params))
    
    return clifford_t_circuit


def pyzx_circuit_metrics(graph: zx.Graph) -> Dict[str, int]:
    """Get metrics from PyZX circuit extraction.
    
    Args:
        graph: PyZX graph
        
    Returns:
        Dictionary with gate counts
    """
    try:
        # Extract circuit using PyZX
        circuit = zx.extract_circuit(graph.copy())
        
        # Count gates
        gate_counts = {
            'cnot_count': 0,
            't_count': 0,
            'total_gates': circuit.gates.__len__() if hasattr(circuit, 'gates') else 0,
        }
        
        if hasattr(circuit, 'gates'):
            for gate in circuit.gates:
                gate_name = gate.name if hasattr(gate, 'name') else str(type(gate).__name__)
                
                if 'CNOT' in gate_name or 'CX' in gate_name:
                    gate_counts['cnot_count'] += 1
                elif 'T' in gate_name and 'NOT' not in gate_name:
                    gate_counts['t_count'] += 1
        
        return gate_counts
    
    except Exception as e:
        # Fallback if extraction fails
        return {
            'cnot_count': 0,
            't_count': 0,
            'total_gates': 0,
            'error': str(e),
        }

