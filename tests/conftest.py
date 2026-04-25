import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Keep CPU-only CI and local tests fast and predictable.
try:
    import torch

    torch.set_num_threads(1)
except Exception:
    pass

# Some helper tests import scripts/preprocess_ptbxl.py, which imports wfdb at
# module import time. The helper functions tested here do not call wfdb, so
# this lightweight stub keeps unit tests runnable in minimal CI/local
# environments where the full ECG waveform dependency is not installed.
try:
    import wfdb  # noqa: F401
except ImportError:
    sys.modules["wfdb"] = types.SimpleNamespace()
