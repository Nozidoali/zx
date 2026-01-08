# ZX Circuit Extraction with GNN

Model-based extraction of circuit-like diagrams from graph-like ZX diagrams.

## Setup

### Option 1: Conda (recommended)
```bash
bash scripts/setup_env.sh
conda activate zx
```

### Option 2: pip
```bash
pip install -r requirements.txt
```

## Execution Order (strict)

```bash
# 1. Baseline
python scripts/run_base.py --verbose

# 2. Dump supervised dataset
python scripts/dump_sup.py --verbose

# 3. Train supervised model
python scripts/train_sup.py --verbose

# 4. Run GNN with supervised checkpoint
python scripts/run_gnn.py --ckpt artifacts/models/sup.pt --output artifacts/gnn_sup.csv --verbose

# 5. Train RL model
python scripts/train_rl.py --init artifacts/models/sup.pt --verbose

# 6. Run GNN with RL checkpoint
python scripts/run_gnn.py --ckpt artifacts/models/rl.pt --output artifacts/gnn_rl.csv --verbose

# 7. Compare results
python scripts/cmp.py

# 8. Run tests
pytest tests/
```

## Structure

```
src/          - all implementation
scripts/      - all entrypoints
artifacts/    - all outputs
tests/        - test suite
```

## Actions

- **CNode**: between frontier pairs
- **Face**: Rz gadget (deg==1, non-trivial angle)  
- **Clifford**: 1q Clifford gates

## Training

- **Supervised**: imitate heuristic trace
- **RL**: REINFORCE + baseline, minimize gate count

