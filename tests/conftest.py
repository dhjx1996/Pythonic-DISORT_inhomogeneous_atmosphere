import sys
from pathlib import Path

# Ensure PythonicDISORT and local helpers are importable regardless of
# whether the package is installed or only present in the source tree.
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(Path(__file__).parent))
