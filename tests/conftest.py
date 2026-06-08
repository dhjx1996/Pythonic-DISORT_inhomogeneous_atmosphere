import sys
from pathlib import Path

import pytest

# Ensure PythonicDISORT and local helpers are importable regardless of
# whether the package is installed or only present in the source tree.
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "float64: stringent convergence/precision test requiring float64. "
        "Excluded from the default (float32) run; execute with "
        "`PYDISORT_RICCATI_JAX_X64=1 pytest -m float64`.",
    )


@pytest.fixture(autouse=True)
def _skip_float64_without_x64(request):
    """A float64-marked test is meaningless (and would crash chasing a
    sub-epsilon tolerance) in the default float32 mode. Skip it cleanly unless
    x64 is actually enabled, so it never runs accidentally."""
    if request.node.get_closest_marker("float64"):
        import jax
        if not jax.config.jax_enable_x64:
            pytest.skip(
                "float64 partition: run with "
                "`PYDISORT_RICCATI_JAX_X64=1 pytest -m float64`"
            )


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    """Clear JAX JIT caches after each test to prevent GPU OOM during long runs."""
    yield
    try:
        import jax
        jax.clear_caches()
    except Exception:
        pass
