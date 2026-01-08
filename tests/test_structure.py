#!/usr/bin/env python3
"""Basic structure tests without external dependencies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing code structure...")

# Test 1: Check file structure
required_files = [
    'src/utils.py',
    'src/bench.py',
    'src/diag.py',
    'src/act.py',
    'src/exp.py',
    'src/qasm.py',
    'src/met.py',
    'src/gnn.py',
    'src/data.py',
    'src/sup.py',
    'src/rl.py',
    'src/infer.py',
    'src/pzx.py',
    'scripts/setup_env.sh',
    'scripts/run_base.py',
    'scripts/dump_sup.py',
    'scripts/train_sup.py',
    'scripts/train_rl.py',
    'scripts/run_gnn.py',
    'scripts/cmp.py',
    'tests/test_all.py',
    '.cursorrules',
    'requirements.txt',
    'README.md',
]

missing = []
for f in required_files:
    if not Path(f).exists():
        missing.append(f)

if missing:
    print(f"ERROR: Missing files: {missing}")
    sys.exit(1)

print("✓ All required files exist")

# Test 2: Check imports without dependencies
print("\nChecking code syntax...")
import ast

for pyfile in Path('src').glob('*.py'):
    try:
        with open(pyfile) as f:
            ast.parse(f.read())
        print(f"✓ {pyfile}")
    except SyntaxError as e:
        print(f"✗ {pyfile}: {e}")
        sys.exit(1)

for pyfile in Path('scripts').glob('*.py'):
    try:
        with open(pyfile) as f:
            ast.parse(f.read())
        print(f"✓ {pyfile}")
    except SyntaxError as e:
        print(f"✗ {pyfile}: {e}")
        sys.exit(1)

# Test 3: Check cursor rules compliance
print("\nChecking cursor rules compliance...")
import re

violations = []

for pyfile in list(Path('src').glob('*.py')) + list(Path('scripts').glob('*.py')):
    with open(pyfile) as f:
        content = f.read()
        lines = content.split('\n')
        
        # Check for excessive comments (allowing docstrings but minimal inline comments)
        comment_lines = [l for l in lines if l.strip().startswith('#')]
        if len(comment_lines) > 5:
            violations.append(f"{pyfile}: Too many comments ({len(comment_lines)})")

if violations:
    print("WARNING: Cursor rule violations:")
    for v in violations:
        print(f"  - {v}")
else:
    print("✓ Code follows cursor rules")

print("\n✅ All basic tests passed!")
print("\nNext steps:")
print("1. Install dependencies: bash scripts/setup_env.sh")
print("2. Run baseline: python scripts/run_base.py")

