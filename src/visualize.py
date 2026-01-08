import pyzx as zx
from pathlib import Path


def save_benchmark_diagram(name: str, graph: zx.Graph, output_dir: str):
    """Save visual diagram of benchmark graph."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tikz_path = f"{output_dir}/{name}.tikz"
    try:
        tikz_code = zx.tikz.to_tikz(graph)
        with open(tikz_path, 'w') as f:
            f.write(tikz_code)
    except:
        pass

    stats_path = f"{output_dir}/{name}_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Benchmark: {name}\n")
        f.write("=" * 60 + "\n\n")

        n_vertices = len(list(graph.vertices()))
        n_edges = len(list(graph.edges()))

        vertex_types = {}
        for v in graph.vertices():
            vtype = str(graph.type(v))
            vertex_types[vtype] = vertex_types.get(vtype, 0) + 1

        f.write(f"Total vertices: {n_vertices}\n")
        f.write(f"Total edges: {n_edges}\n\n")

        f.write("Vertex types:\n")
        for vtype, count in sorted(vertex_types.items()):
            f.write(f"  {vtype:20s}: {count:4d}\n")

        f.write("\nPhase distribution (Z-spiders):\n")
        phases = {}
        for v in graph.vertices():
            if graph.type(v) == zx.VertexType.Z:
                phase = graph.phase(v)
                if abs(phase) > 1e-6:
                    phase_str = f"{phase:.3f}π"
                    phases[phase_str] = phases.get(phase_str, 0) + 1

        if phases:
            for phase, count in sorted(phases.items()):
                f.write(f"  {phase:15s}: {count:4d}\n")
        else:
            f.write("  (no non-trivial phases)\n")

        f.write("\nEdge types:\n")
        edge_types = {}
        for edge in graph.edges():
            etype = str(graph.edge_type(edge))
            edge_types[etype] = edge_types.get(etype, 0) + 1

        for etype, count in sorted(edge_types.items()):
            f.write(f"  {etype:20s}: {count:4d}\n")


def save_all_benchmarks(benchmarks, output_dir: str = "artifacts/benchmarks"):
    """Save diagrams for all benchmarks."""
    for name, graph in benchmarks:
        save_benchmark_diagram(name, graph, output_dir)
