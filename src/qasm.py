import re
from typing import List, Tuple


def circuit_to_qasm(circuit: List[Tuple[str, List[int], List[float]]], num_qubits: int) -> str:
    allowed_gates = {'CZ', 'CNode', 'Rz', 'H', 'S', 'X', 'Z', 'T', 'CNOT'}
    for gate_type, qubits, params in circuit:
        assert gate_type in allowed_gates, f"Invalid gate type: {gate_type}"

    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f'qreg q[{num_qubits}];',
        f'creg c[{num_qubits}];',
        '',
    ]

    for gate_type, qubits, params in circuit:
        qasm_line = _gate_to_qasm(gate_type, qubits, params)
        lines.append(qasm_line)

    return '\n'.join(lines)


def _gate_to_qasm(gate_type: str, qubits: List[int], params: List[float]) -> str:
    if gate_type == 'CZ' or gate_type == 'CNode':
        assert len(qubits) == 2, "CZ/CNode requires 2 qubits"
        return f'cz q[{qubits[0]}],q[{qubits[1]}];'

    elif gate_type == 'CNOT':
        assert len(qubits) == 2, "CNOT requires 2 qubits"
        return f'cx q[{qubits[0]}],q[{qubits[1]}];'

    elif gate_type == 'Rz':
        assert len(qubits) == 1, "Rz requires 1 qubit"
        assert len(params) == 1, "Rz requires 1 parameter"
        angle = params[0]
        return f'rz({angle}) q[{qubits[0]}];'

    elif gate_type == 'H':
        assert len(qubits) == 1, "H requires 1 qubit"
        return f'h q[{qubits[0]}];'

    elif gate_type == 'S':
        assert len(qubits) == 1, "S requires 1 qubit"
        return f's q[{qubits[0]}];'

    elif gate_type == 'T':
        assert len(qubits) == 1, "T requires 1 qubit"
        return f't q[{qubits[0]}];'

    elif gate_type == 'X':
        assert len(qubits) == 1, "X requires 1 qubit"
        return f'x q[{qubits[0]}];'

    elif gate_type == 'Z':
        assert len(qubits) == 1, "Z requires 1 qubit"
        return f'z q[{qubits[0]}];'

    else:
        raise ValueError(f"Unknown gate type: {gate_type}")


def validate_qasm(qasm_str: str) -> bool:
    lines = qasm_str.strip().split('\n')

    if not lines[0].startswith('OPENQASM'):
        return False

    has_qreg = any('qreg' in line for line in lines)
    if not has_qreg:
        return False

    gate_pattern = re.compile(r'^(cz|cx|h|s|t|x|z|rz)\s')
    for line in lines[4:]:
        line = line.strip()
        if line and not line.startswith('//'):
            if not gate_pattern.match(line.lower()):
                if not any(keyword in line for keyword in ['qreg', 'creg', 'include']):
                    return False

    return True
