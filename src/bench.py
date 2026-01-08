import pyzx as zx
from typing import List, Tuple


def get_benchmark_list() -> List[Tuple[str, zx.Graph]]:
    benchmarks = []

    for q in [4, 8]:
        for p_t_int in range(1, 11):
            p_t = p_t_int / 10.0
            for seed in range(10):
                name = f"random_q{q}_g300_pt{p_t:.1f}_s{seed}"
                graph = zx.generate.cliffordT(qubits=q, depth=300, p_t=p_t)
                zx.full_reduce(graph)
                benchmarks.append((name, graph))

    return benchmarks


def load_benchmark(name: str) -> zx.Graph:
    for bench_name, graph in get_benchmark_list():
        if bench_name == name:
            return graph
    assert False, f"Benchmark '{name}' not found"
