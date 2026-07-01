"""Visualization and stability testing for optimization algorithms.

This module provides comprehensive visualization and stability testing tools
for optimization algorithms.

Classes:
    - Visualizer: Visualize optimization algorithm behavior and performance
    - StabilityResults: Store and analyze results from stability tests

Functions:
    - run_stability_test: Run an optimizer multiple times with different seeds
    - compare_optimizers_stability: Compare stability of multiple optimizers
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
