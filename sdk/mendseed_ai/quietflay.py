"""
Backward-compatibility layer for existing MendSeed consumers importing QuietFlay.
Canonical implementation has moved to `quietfray.engine`.

All classes and methods are fully backward-compatible with QuietFlay v0.1.
"""

from __future__ import annotations

import sys
import os

# Ensure src/ is on sys.path if quietfray is not installed globally
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from quietfray.engine import (
    QuietFrayConfig,
    QuietFrayEngine,
    QuietFrayState,
    QuietFlayEngine,
    QuietFlayEngineV01,
    QuietFlayState,
)

__all__ = [
    "QuietFrayConfig",
    "QuietFrayEngine",
    "QuietFrayState",
    "QuietFlayEngine",
    "QuietFlayEngineV01",
    "QuietFlayState",
]
