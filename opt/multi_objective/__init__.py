"""Multi-objective optimization algorithms.

This module provides implementations of multi-objective optimization algorithms
that find Pareto-optimal solutions for problems with multiple competing objectives.

Available Algorithms:
    - AbstractMultiObjectiveOptimizer: Base class for multi-objective optimizers
    - NSGAII: Non-dominated Sorting Genetic Algorithm II

References:
    Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms.
    Wiley, Chichester, UK.
"""

from __future__ import annotations

from opt.multi_objective.abstract_multi_objective import AbstractMultiObjectiveOptimizer
from opt.multi_objective.nsga_ii import NSGAII


__all__ = ["NSGAII", "AbstractMultiObjectiveOptimizer"]
