"""Model-driven circuit extraction inference."""
import time
from typing import Dict, List, Tuple

import torch

from src.gnn import GNNPolicy
from src.diag import DiagramState, from_pyzx_graph
from src.act import generate_candidates, apply_action, Action
from src.utils import log_verbose


def extract_with_policy(model: GNNPolicy,
                       diagram: any,
                       max_steps: int = 1000,
                       temperature: float = 1.0,
                       verbose: bool = False) -> Tuple[List[Tuple[str, List[int], List[float]]], Dict]:
    """Extract circuit using trained policy.
    
    Args:
        model: Trained GNN policy
        diagram: PyZX graph or DiagramState
        max_steps: Maximum extraction steps
        temperature: Sampling temperature
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (circuit, stats)
    """
    # Convert to DiagramState if needed
    if not isinstance(diagram, DiagramState):
        state = from_pyzx_graph(diagram)
    else:
        state = diagram.copy()
    
    model.eval()
    
    stats = {
        'steps': 0,
        'time': 0.0,
        'actions': [],
    }
    
    start_time = time.time()
    
    with torch.no_grad():
        for step in range(max_steps):
            # Check termination
            if state.is_terminated():
                log_verbose(f"Extraction terminated after {step} steps", verbose)
                break
            
            # Generate candidates
            candidates = generate_candidates(state)
            if not candidates:
                log_verbose(f"No candidates available at step {step}", verbose)
                break
            
            # Select action with policy
            try:
                action, log_prob, logits = model.sample_action(state, candidates, temperature)
            except Exception as e:
                log_verbose(f"Action sampling failed: {e}", verbose)
                break
            
            # Apply action
            try:
                apply_action(state, action)
                stats['actions'].append(str(action))
                stats['steps'] += 1
                
                log_verbose(f"Step {step}: {action}", verbose)
                
            except Exception as e:
                log_verbose(f"Action application failed: {e}", verbose)
                break
    
    stats['time'] = time.time() - start_time
    
    return state.circuit, stats


def batch_inference(model: GNNPolicy,
                   benchmarks: List[Tuple[str, any]],
                   max_steps: int = 1000,
                   temperature: float = 1.0,
                   verbose: bool = False) -> List[Tuple[str, List, Dict]]:
    """Run inference on multiple benchmarks.
    
    Args:
        model: Trained GNN policy
        benchmarks: List of (name, graph) tuples
        max_steps: Maximum steps per extraction
        temperature: Sampling temperature
        verbose: Enable verbose logging
        
    Returns:
        List of (name, circuit, stats) tuples
    """
    results = []
    
    for bench_name, graph in benchmarks:
        log_verbose(f"Processing {bench_name}...", verbose)
        
        try:
            circuit, stats = extract_with_policy(
                model, graph, max_steps, temperature, verbose
            )
            results.append((bench_name, circuit, stats))
            
        except Exception as e:
            log_verbose(f"Error processing {bench_name}: {e}", verbose)
            results.append((bench_name, [], {'error': str(e)}))
    
    return results


def greedy_extract(model: GNNPolicy,
                   diagram: any,
                   max_steps: int = 1000,
                   verbose: bool = False) -> Tuple[List[Tuple[str, List[int], List[float]]], Dict]:
    """Extract circuit using greedy action selection.
    
    Args:
        model: Trained GNN policy
        diagram: PyZX graph or DiagramState
        max_steps: Maximum extraction steps
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (circuit, stats)
    """
    # Convert to DiagramState if needed
    if not isinstance(diagram, DiagramState):
        state = from_pyzx_graph(diagram)
    else:
        state = diagram.copy()
    
    model.eval()
    
    stats = {
        'steps': 0,
        'time': 0.0,
        'actions': [],
    }
    
    start_time = time.time()
    
    with torch.no_grad():
        for step in range(max_steps):
            # Check termination
            if state.is_terminated():
                break
            
            # Generate candidates
            candidates = generate_candidates(state)
            if not candidates:
                break
            
            # Greedy selection
            try:
                logits = model(state, candidates)
                action_idx = torch.argmax(logits).item()
                action = candidates[action_idx]
            except Exception as e:
                log_verbose(f"Action selection failed: {e}", verbose)
                break
            
            # Apply action
            try:
                apply_action(state, action)
                stats['actions'].append(str(action))
                stats['steps'] += 1
                
            except Exception as e:
                log_verbose(f"Action application failed: {e}", verbose)
                break
    
    stats['time'] = time.time() - start_time
    
    return state.circuit, stats

