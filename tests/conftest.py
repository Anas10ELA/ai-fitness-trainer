"""
tests/conftest.py — shared pytest fixtures
"""
import os
import sys
from pathlib import Path

# Make root importable from tests/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Minimal env for config validation
os.environ.setdefault("WORKERS", "1")
os.environ.setdefault("LOG_LEVEL", "WARNING")
