"""Constrained optimization algorithms.

This module contains optimizers specifically designed for handling optimization problems
with equality and/or inequality constraints. Includes: Augmented Lagrangian Method,
Successive Linear Programming, Penalty Method, Barrier Method (Interior Point),
and Sequential Quadratic Programming.
"""

from __future__ import annotations

from opt.constrained.augmented_lagrangian_method import AugmentedLagrangian
from opt.constrained.barrier_method import BarrierMethodOptimizer
from opt.constrained.penalty_method import PenaltyMethodOptimizer
from opt.constrained.sequential_quadratic_programming import (
    SequentialQuadraticProgramming,
)
from opt.constrained.successive_linear_programming import SuccessiveLinearProgramming


__all__: list[str] = [
    "AugmentedLagrangian",
    "BarrierMethodOptimizer",
    "PenaltyMethodOptimizer",
    "SequentialQuadraticProgramming",
    "SuccessiveLinearProgramming",
]
