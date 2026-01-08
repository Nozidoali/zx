import json
from typing import Any, Dict

import pyzx as zx
from src.diag import DiagramState
from src.act import Action


def export_graph_state(state: DiagramState, path: str) -> None:
    graph_data = {
        'nodes': [],
        'edges': [],
        'frontier': list(state.frontier),
        'num_qubits': state.num_qubits,
    }

    for v in state.graph.vertices():
        node_data = {
            'id': int(v),
            'type': str(state.graph.type(v)),
            'phase': float(state.graph.phase(v)),
            'row': int(state.graph.row(v)) if hasattr(state.graph, 'row') else 0,
            'qubit': int(state.graph.qubit(v)) if hasattr(state.graph, 'qubit') else 0,
        }
        graph_data['nodes'].append(node_data)

    for edge in state.graph.edges():
        s, t = edge
        edge_type = state.graph.edge_type(edge)
        graph_data['edges'].append({
            'source': int(s),
            'target': int(t),
            'type': str(edge_type),
        })

    with open(path, 'w') as f:
        json.dump(graph_data, f, indent=2)


def dump_trace_step(state: DiagramState, action: Action, step_id: int, trace_file: str) -> None:
    step_data = {
        'step': step_id,
        'action': {
            'type': action.__class__.__name__,
            'params': _action_to_dict(action),
        },
        'frontier': list(state.frontier),
        'num_nodes': len(list(state.graph.vertices())),
        'num_edges': len(list(state.graph.edges())),
    }

    with open(trace_file, 'a') as f:
        f.write(json.dumps(step_data) + '\n')


def _action_to_dict(action: Action) -> Dict[str, Any]:
    if hasattr(action, '__dict__'):
        return action.__dict__
    return {}
