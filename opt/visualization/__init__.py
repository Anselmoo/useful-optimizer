"""Visualization and stability testing for optimization algorithms.

This module provides comprehensive visualization and stability testing tools
for optimization algorithms.

Classes:
    - Visualizer: Visualize optimization algorithm behavior and performance
    - StabilityResults: Store and analyze results from stability tests

Functions:
    - run_stability_test: Run an optimizer multiple times with different seeds
    - compare_optimizers_stability: Compare stability of multiple optimizers

Example:
    >>> from opt.swarm_intelligence.particle_swarm import ParticleSwarm
    >>> from opt.benchmark.functions import shifted_ackley
    >>> from opt.visualization import Visualizer, run_stability_test
    >>>
    >>> # Single run with visualization
    >>> pso = ParticleSwarm(
    ...     func=shifted_ackley,
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=2,
    ...     max_iter=100,
    ...     track_history=True
    ... )
    >>> best_solution, best_fitness = pso.search()
    >>>
    >>> viz = Visualizer(pso)
    >>> viz.plot_convergence()
    >>> viz.plot_trajectory()
    >>> viz.plot_average_fitness()
    >>>
    >>> # Stability test with multiple seeds
    >>> results = run_stability_test(
    ...     optimizer_class=ParticleSwarm,
    ...     func=shifted_ackley,
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=2,
    ...     max_iter=100,
    ...     seeds=[42, 123, 456, 789, 1011],
    ... )
    >>> results.print_summary()
    >>> results.plot_boxplot()
"""

from __future__ import annotations

from opt.visualization.stability import StabilityResults
from opt.visualization.stability import compare_optimizers_stability
from opt.visualization.stability import run_stability_test
from opt.visualization.visualizer import Visualizer


__all__ = [
    "StabilityResults",
    "Visualizer",
    "compare_optimizers_stability",
    "run_stability_test",
]
