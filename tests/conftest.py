import sys
from pathlib import Path

import pytest

# Ensure PythonicDISORT and local helpers are importable regardless of
# whether the package is installed or only present in the source tree.
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    """Clear JAX JIT caches after each test to prevent GPU OOM during long runs."""
    yield
    try:
        import jax
        jax.clear_caches()
    except Exception:
        pass
