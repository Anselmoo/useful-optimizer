"""Penalty Method Optimizer.

This module implements the Penalty Method for constrained optimization,
transforming constrained problems into unconstrained ones.

The algorithm adds penalty terms for constraint violations to the objective
function, with increasing penalty coefficients over iterations.

Reference:
    Nocedal, J., & Wright, S. J. (2006).
    Numerical Optimization (2nd ed.).
    Springer. Chapter 17: Penalty and Augmented Lagrangian Methods.

Example:
    >>> from opt.benchmark.functions import sphere
    >>> # Minimize sphere with constraint sum(x) >= 0
    >>> def constraint(x):
    ...     return -np.sum(x)  # g(x) <= 0 form
    >>> optimizer = PenaltyMethodOptimizer(
    ...     func=sphere,
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=2,
    ...     constraints=[constraint],
    ...     max_iter=100
    ... )
    >>> best_solution, best_fitness = optimizer.search()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import minimize

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class PenaltyMethodOptimizer(AbstractOptimizer):
    """Penalty Method for constrained optimization.

    This algorithm:
    1. Converts constraints to penalty terms
    2. Progressively increases penalty coefficients
    3. Uses gradient-based optimization on penalized objective

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        constraints: List of constraint functions (g(x) <= 0 form).
        eq_constraints: List of equality constraints (h(x) = 0 form).
        max_iter: Maximum outer iterations.
        initial_penalty: Starting penalty coefficient.
        penalty_growth: Penalty coefficient growth factor.


    Example:
        >>> from opt.constrained.penalty_method import PenaltyMethodOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = PenaltyMethodOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = PenaltyMethodOptimizer(
        ...     func=shifted_ackley,
        ...     dim=2,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     max_iter=10
        ... )
        >>> _, fitness = optimizer.search()
        >>> isinstance(float(fitness), float)
        True
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        constraints: list[Callable[[np.ndarray], float]] | None = None,
        eq_constraints: list[Callable[[np.ndarray], float]] | None = None,
        max_iter: int = 100,
        initial_penalty: float = 1.0,
        penalty_growth: float = 2.0,
    ) -> None:
        """Initialize Penalty Method Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            constraints: Inequality constraints g(x) <= 0. Defaults to None.
            eq_constraints: Equality constraints h(x) = 0. Defaults to None.
            max_iter: Outer iterations. Defaults to 100.
            initial_penalty: Starting penalty. Defaults to 1.0.
            penalty_growth: Penalty growth rate. Defaults to 2.0.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.constraints = constraints or []
        self.eq_constraints = eq_constraints or []
        self.initial_penalty = initial_penalty
        self.penalty_growth = penalty_growth

    def _penalized_objective(self, x: np.ndarray, penalty: float) -> float:
        """Compute penalized objective function.

        Args:
            x: Point to evaluate.
            penalty: Current penalty coefficient.

        Returns:
            Penalized objective value.
        """
        obj = self.func(x)

        # Inequality constraints: penalty for g(x) > 0
        for g in self.constraints:
            violation = max(0, g(x))
            obj += penalty * violation**2

        # Equality constraints: penalty for h(x) != 0
        for h in self.eq_constraints:
            violation = h(x)
            obj += penalty * violation**2

        return obj

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Penalty Method optimization.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize from random point
        current = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        bounds = [(self.lower_bound, self.upper_bound)] * self.dim
        penalty = self.initial_penalty

        best_solution = current.copy()
        best_fitness = self.func(current)
        best_violation = self._compute_violation(current)

        for _ in range(self.max_iter):
            # Minimize penalized objective
            result = minimize(
                lambda x: self._penalized_objective(x, penalty),
                current,
                method="L-BFGS-B",
                bounds=bounds,
            )
            current = result.x

            # Compute actual fitness and constraint violation
            fitness = self.func(current)
            violation = self._compute_violation(current)

            # Update best if feasible or less violated
            if violation < best_violation or (
                violation <= 1e-6 and fitness < best_fitness
            ):
                best_solution = current.copy()
                best_fitness = fitness
                best_violation = violation

            # Increase penalty
            penalty *= self.penalty_growth

            # Early termination if constraints satisfied
            if violation < 1e-8:
                break

        return best_solution, best_fitness

    def _compute_violation(self, x: np.ndarray) -> float:
        """Compute total constraint violation.

        Args:
            x: Point to evaluate.

        Returns:
            Total violation measure.
        """
        violation = 0.0

        for g in self.constraints:
            violation += max(0, g(x)) ** 2

        for h in self.eq_constraints:
            violation += h(x) ** 2

        return np.sqrt(violation)


if __name__ == "__main__":
    from opt.benchmark.functions import sphere

    # Simple constraint: sum(x) >= 1 (i.e., -sum(x) + 1 <= 0)
    def constraint(x: np.ndarray) -> float:
        return -np.sum(x) + 1

    optimizer = PenaltyMethodOptimizer(
        func=sphere,
        lower_bound=-5,
        upper_bound=5,
        dim=2,
        constraints=[constraint],
        max_iter=100,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
    print(f"Constraint satisfied: {np.sum(best_solution) >= 1}")
