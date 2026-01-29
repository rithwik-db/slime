"""Pytest configuration for Slime tests."""

import sys
from pathlib import Path

# Add slime package to path
slime_root = Path(__file__).parent.parent
sys.path.insert(0, str(slime_root))
