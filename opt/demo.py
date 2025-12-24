"""Centralized demo runner for optimizer demonstrations.

This module provides a standardized way to run demonstrations for any optimizer,
ensuring consistent output formatting and reducing code duplication across the
codebase.

Example:
    >>> from opt.demo import run_demo
    >>> from opt.swarm_intelligence.particle_swarm import ParticleSwarm
    >>> solution, fitness = run_demo(ParticleSwarm)
    Running ParticleSwarm demo...
      Function: shifted_ackley
      Dimensions: 2
      Bounds: [-2.768, 2.768]
      Max iterations: 100
    <BLANKLINE>
    Results:
      Best solution found: [...]
      Best fitness value: ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from opt.abstract import AbstractOptimizer


def run_demo(
    optimizer_class: type[AbstractOptimizer],
    *,
    func: Callable[[np.ndarray], float] = shifted_ackley,
    dim: int = 2,
    lower_bound: float = -2.768,
    upper_bound: float = 2.768,
    max_iter: int = 100,
    **kwargs: Any,
) -> tuple[np.ndarray, float]:
    """Run a standardized demo for any optimizer.

    Args:
        optimizer_class: The optimizer class to demonstrate.
        func: Benchmark function to optimize. Defaults to shifted_ackley.
        dim: Dimensionality of the search space. Defaults to 2.
        lower_bound: Lower bound of the search space. Defaults to -2.768.
        upper_bound: Upper bound of the search space. Defaults to 2.768.
        max_iter: Maximum iterations. Defaults to 100.
        **kwargs: Additional optimizer-specific parameters.

    Returns:
        Tuple of (best_solution, best_fitness).

    Example:
        >>> from opt.demo import run_demo
        >>> from opt.swarm_intelligence.particle_swarm import ParticleSwarm
        >>> solution, fitness = run_demo(ParticleSwarm)
        Running ParticleSwarm demo...
          Function: shifted_ackley
          Dimensions: 2
          Bounds: [-2.768, 2.768]
          Max iterations: 100
        <BLANKLINE>
        Results:
          Best solution found: [...]
          Best fitness value: ...
    """
    print(f"Running {optimizer_class.__name__} demo...")
    print(f"  Function: {func.__name__}")
    print(f"  Dimensions: {dim}")
    print(f"  Bounds: [{lower_bound}, {upper_bound}]")
    print(f"  Max iterations: {max_iter}")

    optimizer = optimizer_class(
        func=func,
        dim=dim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        max_iter=max_iter,
        **kwargs,
    )

    best_solution, best_fitness = optimizer.search()

    print("\nResults:")
    print(f"  Best solution found: {best_solution}")
    print(f"  Best fitness value: {best_fitness:.6f}")

    return best_solution, best_fitness
