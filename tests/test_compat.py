"""Tests for compatibility patch behavior."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock


def test_pkg_resources_patch_only_when_unavailable(monkeypatch):
    """Compat should avoid shadowing real pkg_resources installs."""
    import quantum_optical_bus.compat as compat

    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)
    try:
        importlib.import_module("pkg_resources")
    except ImportError:
        pkg_resources_available = False
    else:
        pkg_resources_available = True
        monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)

    importlib.reload(compat)

    patched = sys.modules.get("pkg_resources")
    assert patched is not None

    if not pkg_resources_available:
        assert isinstance(patched, MagicMock)
    else:
        assert not isinstance(patched, MagicMock)
