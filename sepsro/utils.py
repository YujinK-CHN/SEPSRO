"""Small helpers shared across the package."""

import random

import numpy as np


def set_seed(seed):
    """Seed Python, NumPy and (if installed) PyTorch.

    Methods should still draw from the np.random.Generator main.py hands them
    rather than the global NumPy state; this only covers libraries that use
    global RNGs internally.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
