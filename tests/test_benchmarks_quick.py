"""Quick benchmark sanity tests for representative optimizers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from opt.benchmark.functions import sphere
from opt.probabilistic.sequential_monte_carlo import SequentialMonteCarloOptimizer
from opt.swarm_intelligence.particle_swarm import ParticleSwarm


@pytest.mark.parametrize(
    ("cls", "params"),
    [
        (
            SequentialMonteCarloOptimizer,
            {
                "func": sphere,
                "lower_bound": -5,
                "upper_bound": 5,
                "dim": 10,
                "max_iter": 50,
                "seed": 42,
                "track_history": True,
            },
        ),
        (
            ParticleSwarm,
            {
                "func": sphere,
                "lower_bound": -5,
                "upper_bound": 5,
                "dim": 10,
                "max_iter": 50,
                "seed": 42,
                "track_history": True,
            },
        ),
    ],
)
def test_optimizer_quick_example_sanity(cls: Any, params: dict[str, Any]) -> None:
    """Quick sanity: shape, type, reproducibility, and history presence."""
    opt1 = cls(**params)
    solution, fitness = opt1.search()

    # Basic structure checks
    assert isinstance(solution, np.ndarray)
    assert solution.shape == (params["dim"],)
    assert isinstance(fitness, float)

    # Reproducibility
    opt2 = cls(**params)
    s2, f2 = opt2.search()
    assert np.allclose(solution, s2, atol=1e-12)
    assert np.allclose(fitness, f2, atol=1e-12)

    # History minimality
    if getattr(opt1, "history", None) is not None:
        assert "best_fitness" in opt1.history
        assert isinstance(opt1.history["best_fitness"], list)


# Use marker to allow running full tests separately in CI/nightly
@pytest.mark.benchmark_full
def test_optimizer_full_sphere_example() -> None:
    """Full-run example test scaffold."""
    _solution, fitness = SequentialMonteCarloOptimizer(
        func=sphere,
        lower_bound=-5,
        upper_bound=5,
        dim=10,
        max_iter=2000,
        seed=0,
        track_history=True,
    ).search()
    # Conservative full-run threshold for sphere dim=10
    assert fitness < 1.0

    # Save a minimal artifact for the test (in CI this should upload)
    opt = SequentialMonteCarloOptimizer(
        func=sphere,
        lower_bound=-5,
        upper_bound=5,
        dim=10,
        max_iter=2000,
        seed=0,
        track_history=True,
    )
    _s, _f = opt.search()
    assert getattr(opt, "history", None) is not None
