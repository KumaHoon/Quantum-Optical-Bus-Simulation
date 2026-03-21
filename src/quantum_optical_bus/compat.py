"""Compatibility shims for optional runtime dependencies.

This module must be imported before any Strawberry Fields or Meep usage.

Only a stable fallback for ``pkg_resources`` is retained here.
The project intentionally declares a supported SciPy range in ``pyproject.toml`` to
avoid runtime monkey-patching for removed APIs (for example, ``scipy.integrate.simps``).
"""

import sys
from unittest.mock import MagicMock

# Patch 1: Provide pkg_resources fallback only when import is unavailable.
try:
    import pkg_resources  # noqa: F401
except ImportError:
    sys.modules.setdefault("pkg_resources", MagicMock())
