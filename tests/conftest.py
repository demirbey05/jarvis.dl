# tests/conftest.py
import sys
from pathlib import Path

# add project root (one level up from tests/) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))