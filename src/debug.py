import json
import pyzx as zx
from src.diag import DiagramState
from src.act import Action


def save_graph_debug(state: DiagramState, path: str):
    """Save graph in human-readable format for debugging."""
    debug_info = {
        'num_qubits': state.num_qubits,
        'num_vertices': len(list(state.graph.vertices())),
        'num_edges': len(list(state.graph.edges())),
        'frontier': sorted(list(state.frontier)),
        'vertices': [],
        'edges': [],
        'circuit_so_far': state.circuit,
    }

    for v in state.graph.vertices():
        vtype = state.graph.type(v)
        vertex_info = {
            'id': int(v),
            'type': str(vtype),
            'phase': float(state.graph.phase(v)),
            'degree': state.graph.vertex_degree(v),
            'in_frontier': v in state.frontier,
        }

        if hasattr(state.graph, 'row'):
            vertex_info['row'] = int(state.graph.row(v))
        if hasattr(state.graph, 'qubit'):
            vertex_info['qubit'] = int(state.graph.qubit(v))

        debug_info['vertices'].append(vertex_info)

    for edge in state.graph.edges():
        s, t = edge
        edge_type = state.graph.edge_type(edge)
        debug_info['edges'].append({
            'source': int(s),
            'target': int(t),
            'type': str(edge_type),
        })

    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=2)


def save_action_debug(action: Action, path: str, append: bool = True):
    """Save action in human-readable format."""
    action_info = {
        'type': action.__class__.__name__,
        'details': str(action),
        'params': action.__dict__ if hasattr(action, '__dict__') else {},
    }

    mode = 'a' if append else 'w'
    with open(path, mode) as f:
        f.write(json.dumps(action_info, indent=2) + '\n\n')


def save_step_debug(step_num: int, state: DiagramState, action: Action,
                    candidates: list, label_idx: int, base_path: str):
    debug_path = f"{base_path}_step{step_num:04d}.txt"

    with open(debug_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"STEP {step_num}\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"GRAPH STATE:\n")
        f.write(f"  Qubits: {state.num_qubits}\n")
        f.write(f"  Vertices: {len(list(state.graph.vertices()))}\n")
        f.write(f"  Edges: {len(list(state.graph.edges()))}\n")
        f.write(f"  Frontier size: {len(state.frontier)}\n")
        f.write(f"  Frontier nodes: {sorted(list(state.frontier))}\n\n")

        f.write(f"VERTICES:\n")
        for v in sorted(state.graph.vertices()):
            vtype = state.graph.type(v)
            phase = state.graph.phase(v)
            degree = state.graph.vertex_degree(v)
            in_frontier = "✓" if v in state.frontier else " "
            f.write(f"  [{in_frontier}] Node {v:3d}: {str(vtype):20s} "
                   f"phase={phase:6.3f}, deg={degree}\n")

        f.write(f"\nEDGES:\n")
        for edge in state.graph.edges():
            s, t = edge
            etype = state.graph.edge_type(edge)
            f.write(f"  {s} <--{str(etype)}--> {t}\n")

        f.write(f"\n{'='*80}\n")
        f.write(f"CANDIDATE ACTIONS ({len(candidates)} total):\n")
        f.write(f"{'='*80}\n\n")

        for i, cand in enumerate(candidates):
            marker = ">>> CHOSEN <<<" if i == label_idx else ""
            f.write(f"  [{i:3d}] {str(cand):60s} {marker}\n")

        f.write(f"\n{'='*80}\n")
        f.write(f"SELECTED ACTION (index {label_idx}):\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"  Type: {action.__class__.__name__}\n")
        f.write(f"  Details: {str(action)}\n")
        f.write(f"  Params: {action.__dict__ if hasattr(action, '__dict__') else {}}\n")

        f.write(f"\n{'='*80}\n")
        f.write(f"CIRCUIT SO FAR ({len(state.circuit)} gates):\n")
        f.write(f"{'='*80}\n\n")
        for gate_type, qubits, params in state.circuit:
            params_str = f", params={params}" if params else ""
            f.write(f"  {gate_type}(qubits={qubits}{params_str})\n")

        f.write(f"\n")


def create_debug_summary(benchmark_name: str, steps_data: list, output_dir: str):
    """Create a summary file for the entire extraction sequence."""
    summary_path = f"{output_dir}/{benchmark_name}_SUMMARY.txt"

    with open(summary_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"EXTRACTION SUMMARY: {benchmark_name}\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"Total steps: {len(steps_data)}\n\n")

        action_counts = {}
        for step_data in steps_data:
            action_type = step_data['action_type']
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        f.write(f"ACTION STATISTICS:\n")
        for action_type, count in sorted(action_counts.items()):
            f.write(f"  {action_type:20s}: {count:4d} times\n")

        f.write(f"\n{'='*80}\n")
        f.write(f"STEP-BY-STEP SEQUENCE:\n")
        f.write(f"{'='*80}\n\n")

        for i, step_data in enumerate(steps_data):
            f.write(f"Step {i:4d}: {step_data['action_str']}\n")
