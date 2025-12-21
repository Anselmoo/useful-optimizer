"""Multi-objective optimization algorithms.

This module provides implementations of multi-objective optimization algorithms
that find Pareto-optimal solutions for problems with multiple competing objectives.

Available Algorithms:
    - AbstractMultiObjectiveOptimizer: Base class for multi-objective optimizers

References:
    Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms.
    Wiley, Chichester, UK.
"""

from __future__ import annotations

from opt.multi_objective.abstract_multi_objective import AbstractMultiObjectiveOptimizer


__all__ = ["AbstractMultiObjectiveOptimizer"]
