from __future__ import annotations

import pytest

from opt.benchmark.functions import sphere
from opt.probabilistic.sequential_monte_carlo import SequentialMonteCarloOptimizer


@pytest.mark.benchmark_full
def test_some_optimizer_full():
    """Scaffold for nightly full-run benchmark tests.

    This test runs a larger budget and validates conservative numeric thresholds.
    """
    opt = SequentialMonteCarloOptimizer(
        func=sphere,
        lower_bound=-5,
        upper_bound=5,
        dim=10,
        max_iter=2000,
        seed=0,
        track_history=True,
    )
    _solution, fitness = opt.search()
    assert fitness < 1.0
    # Save a minimal artifact for the test (in CI this should upload)
    _ = opt.search()
    assert getattr(opt, "history", None) is not None
