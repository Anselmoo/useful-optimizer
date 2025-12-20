"""Constrained optimization algorithms.

This module contains optimizers specifically designed for handling optimization problems
with equality and/or inequality constraints. Includes: Augmented Lagrangian Method
and Successive Linear Programming.
"""

from __future__ import annotations

from opt.constrained.augmented_lagrangian_method import AugmentedLagrangian
from opt.constrained.successive_linear_programming import SuccessiveLinearProgramming


__all__: list[str] = ["AugmentedLagrangian", "SuccessiveLinearProgramming"]
