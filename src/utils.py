import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def validate_file_exists(path: str, desc: str = "File"):
    assert os.path.exists(path) and os.path.getsize(path) > 0, f"{desc} missing or empty: {path}"


def log_verbose(msg: str, verbose: bool = False):
    if verbose:
        print(msg)


def fail(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)
