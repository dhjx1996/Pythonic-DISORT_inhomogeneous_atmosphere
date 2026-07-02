import sys
from pathlib import Path

# The HPC gates exercise the production worker entry points.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
