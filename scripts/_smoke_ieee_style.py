"""Smoke test for the shared IEEE visualization style helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from quantum_optical_bus.viz_style_ieee import (
    apply_ieee_axes,
    ieee_figsize,
    save_ieee,
    set_ieee_rcparams,
)


def main() -> None:
    """Create a trivial styled figure and write it to a temporary file."""

    set_ieee_rcparams()
    fig, ax = plt.subplots(figsize=ieee_figsize())

    x = [0, 1, 2, 3]
    y = [0.0, 1.0, 0.6, 1.2]
    ax.plot(x, y, marker="o", label="demo")
    ax.legend()
    apply_ieee_axes(ax, xlabel="time (a.u.)", ylabel="signal", title="IEEE style smoke")

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "ieee_smoke.png"
        save_ieee(fig, out, dpi=150)
        assert out.exists()

    plt.close(fig)


if __name__ == "__main__":
    main()
