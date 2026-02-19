"""Backward-compatible import shim for visualization styling helpers.

The canonical implementation now lives in :mod:`quantum_optical_bus.viz_style_ieee`.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantum_optical_bus.viz_style_ieee import *  # noqa: E402, F401, F403
