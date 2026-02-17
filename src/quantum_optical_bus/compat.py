"""
Compatibility patches for dependency issues.

This module must be imported before any Strawberry Fields or Meep usage.
It patches known issues with pkg_resources and scipy.integrate.simps
in newer Python / SciPy versions.
"""

import sys
from unittest.mock import MagicMock

# Patch 1: Provide pkg_resources fallback only when import is unavailable.
try:
    import pkg_resources  # noqa: F401
except ImportError:
    sys.modules.setdefault("pkg_resources", MagicMock())

# Patch 2: Fix scipy.integrate.simps removal (removed in SciPy 1.14+)
import scipy.integrate

if not hasattr(scipy.integrate, "simps"):
    if hasattr(scipy.integrate, "simpson"):
        scipy.integrate.simps = scipy.integrate.simpson
