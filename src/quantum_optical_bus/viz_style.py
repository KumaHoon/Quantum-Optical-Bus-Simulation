"""Compatibility wrapper over shared IEEE plotting helpers."""

from __future__ import annotations

from matplotlib.figure import Figure
from quantum_optical_bus import viz_style_ieee as _core

for _name in _core.__all__:
    globals()[_name] = getattr(_core, _name)

Figure = Figure

__all__ = list(_core.__all__) + ["Figure"]
