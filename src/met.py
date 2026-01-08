from typing import Dict, List, Tuple

import pyzx as zx


def count_gates(circuit: List[Tuple[str, List[int], List[float]]]) -> Dict[str, int]:
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
    clifford_t_circuit = []

    for gate_type, qubits, params in circuit:
        if gate_type in ['H', 'S', 'T', 'CNOT', 'CX', 'X', 'Z']:
            clifford_t_circuit.append((gate_type, qubits, params))

        elif gate_type in ['CZ', 'CNode']:
            clifford_t_circuit.append(('H', [qubits[1]], []))
            clifford_t_circuit.append(('CNOT', qubits, []))
            clifford_t_circuit.append(('H', [qubits[1]], []))

        elif gate_type == 'Rz':
            angle = params[0] if params else 0

            import math
            t_angle = math.pi / 4

            if abs(angle - t_angle) < 0.1:
                clifford_t_circuit.append(('T', qubits, []))
            elif abs(angle + t_angle) < 0.1:
                clifford_t_circuit.append(('S', qubits, []))
                clifford_t_circuit.append(('T', qubits, []))
                clifford_t_circuit.append(('S', qubits, []))
            else:
                num_t = max(1, int(abs(angle) / t_angle))
                for _ in range(num_t):
                    clifford_t_circuit.append(('T', qubits, []))

        else:
            clifford_t_circuit.append((gate_type, qubits, params))

    return clifford_t_circuit


def pyzx_circuit_metrics(graph: zx.Graph) -> Dict[str, int]:
    try:
        circuit = zx.extract_circuit(graph.copy())

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
        return {
            'cnot_count': 0,
            't_count': 0,
            'total_gates': 0,
            'error': str(e),
        }
