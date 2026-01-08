#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bench import get_benchmark_list
from src.pzx import run_pyzx_with_trace
from src.data import save_supervised_data, load_traces_from_jsonl
from src.diag import from_pyzx_graph
from src.act import generate_candidates
from src.utils import set_seed, ensure_dir, validate_file_exists, log_verbose
from src.debug import save_step_debug, create_debug_summary


def main():
    parser = argparse.ArgumentParser(description='Dump supervised dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='artifacts/data/sup_train.npz')
    parser.add_argument('--val-output', type=str, default='artifacts/data/sup_val.npz')
    parser.add_argument('--trace-dir', type=str, default='artifacts/traces')
    parser.add_argument('--debug-dir', type=str, default='artifacts/debug')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--debug', action='store_true', help='Save human-readable debug files')
    parser.add_argument('--debug-limit', type=int, default=3, help='Number of benchmarks to save debug info for')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    set_seed(args.seed)
    ensure_dir(args.trace_dir)
    ensure_dir(str(Path(args.output).parent))
    if args.debug:
        ensure_dir(args.debug_dir)
    
    benchmarks = get_benchmark_list()
    log_verbose(f"Generating traces for {len(benchmarks)} benchmarks", args.verbose)
    
    states_all = []
    candidates_all = []
    labels_all = []
    
    for bench_idx, (bench_name, graph) in enumerate(benchmarks):
        trace_file = f"{args.trace_dir}/{bench_name}.jsonl"
        open(trace_file, 'w').close()
        
        log_verbose(f"Tracing {bench_name}...", args.verbose)
        run_pyzx_with_trace(graph, trace_file, args.verbose)
        
        traces = load_traces_from_jsonl(trace_file)
        
        state = from_pyzx_graph(graph)
        
        save_debug = args.debug and bench_idx < args.debug_limit
        debug_steps = []
        
        for step_idx, trace_step in enumerate(traces):
            candidates = generate_candidates(state)
            if not candidates:
                break
            
            action_params = trace_step['action']['params']
            label = 0
            for i, cand in enumerate(candidates):
                if cand.__dict__ == action_params:
                    label = i
                    break
            
            if save_debug:
                bench_debug_dir = f"{args.debug_dir}/{bench_name}"
                ensure_dir(bench_debug_dir)
                
                action = candidates[label]
                save_step_debug(step_idx, state, action, candidates, label, 
                              f"{bench_debug_dir}/{bench_name}")
                
                debug_steps.append({
                    'action_type': action.__class__.__name__,
                    'action_str': str(action),
                })
            
            states_all.append(state.copy())
            candidates_all.append(candidates)
            labels_all.append(label)
        
        if save_debug and debug_steps:
            create_debug_summary(bench_name, debug_steps, f"{args.debug_dir}/{bench_name}")
            log_verbose(f"  Debug files saved to {args.debug_dir}/{bench_name}/", args.verbose)
    
    n_val = int(len(states_all) * args.val_split)
    indices = list(range(len(states_all)))
    import random
    random.shuffle(indices)
    
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    train_states = [states_all[i] for i in train_idx]
    train_candidates = [candidates_all[i] for i in train_idx]
    train_labels = [labels_all[i] for i in train_idx]
    
    val_states = [states_all[i] for i in val_idx]
    val_candidates = [candidates_all[i] for i in val_idx]
    val_labels = [labels_all[i] for i in val_idx]
    
    save_supervised_data(train_states, train_candidates, train_labels, args.output)
    save_supervised_data(val_states, val_candidates, val_labels, args.val_output)
    
    validate_file_exists(args.output, "Train dataset")
    validate_file_exists(args.val_output, "Val dataset")
    
    print(f"Dataset saved: {len(train_states)} train, {len(val_states)} val")
    if args.debug:
        print(f"Debug files saved for first {args.debug_limit} benchmarks in {args.debug_dir}/")
    
    sys.exit(0)


if __name__ == '__main__':
    main()

