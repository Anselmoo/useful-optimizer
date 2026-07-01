"""Tests for the centralized demo module."""

from __future__ import annotations

from opt.benchmark.functions import rosenbrock
from opt.benchmark.functions import sphere
from opt.demo import run_demo
from opt.swarm_intelligence.particle_swarm import ParticleSwarm


def test_run_demo_basic():
    """Test basic demo functionality."""
    best_solution, best_fitness = run_demo(ParticleSwarm, max_iter=10)

    # Check return types
    assert best_solution is not None
    assert isinstance(best_fitness, float)

    # Check solution dimensions
    assert len(best_solution) == 2


def test_run_demo_custom_function():
    """Test demo with custom function."""
    best_solution, best_fitness = run_demo(ParticleSwarm, func=sphere, max_iter=10)

    assert best_solution is not None
    assert isinstance(best_fitness, float)


def test_run_demo_custom_bounds():
    """Test demo with custom bounds."""
    best_solution, best_fitness = run_demo(
        ParticleSwarm,
        func=rosenbrock,
        lower_bound=-5.0,
        upper_bound=5.0,
        dim=3,
        max_iter=10,
    )

    assert len(best_solution) == 3
    assert isinstance(best_fitness, float)


def test_run_demo_with_kwargs():
    """Test demo with additional optimizer kwargs."""
    best_solution, best_fitness = run_demo(
        ParticleSwarm, max_iter=10, population_size=20, c1=2.0, c2=2.0
    )

    assert best_solution is not None
    assert isinstance(best_fitness, float)
