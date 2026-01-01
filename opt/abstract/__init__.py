"""Abstract base classes for optimization algorithms.

This module provides base classes and utilities for single-objective
and multi-objective optimization algorithms.
"""

from __future__ import annotations

from opt.abstract.history import HistoryConfig
from opt.abstract.history import OptimizationHistory
from opt.abstract.multi_objective import AbstractMultiObjectiveOptimizer
from opt.abstract.single_objective import AbstractOptimizer


__all__ = [
    "AbstractMultiObjectiveOptimizer",
    "AbstractOptimizer",
    "HistoryConfig",
    "OptimizationHistory",
]
