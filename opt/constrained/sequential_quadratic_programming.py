"""Sequential Quadratic Programming Optimizer.

This module implements the Sequential Quadratic Programming (SQP) algorithm,
a powerful method for solving nonlinear constrained optimization problems.

The algorithm iteratively solves quadratic programming subproblems to
approximate the original nonlinear problem.

Reference:
    Nocedal, J., & Wright, S. J. (2006).
    Numerical Optimization (2nd ed.).
    Springer. Chapter 18: Sequential Quadratic Programming.

Example:
    >>> from opt.benchmark.functions import sphere
    >>> # Minimize sphere with constraint sum(x) = 1
    >>> def eq_constraint(x):
    ...     return np.sum(x) - 1
    >>> optimizer = SequentialQuadraticProgramming(
    ...     func=sphere,
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=2,
    ...     eq_constraints=[eq_constraint],
    ...     max_iter=100,
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


class SequentialQuadraticProgramming(AbstractOptimizer):
    """Sequential Quadratic Programming for constrained optimization.

    This algorithm:
    1. Approximates the Lagrangian Hessian using BFGS updates
    2. Solves QP subproblems at each iteration
    3. Uses merit function for step acceptance

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        constraints: List of inequality constraint functions (g(x) <= 0).
        eq_constraints: List of equality constraint functions (h(x) = 0).
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.
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
        tol: float = 1e-6,
    ) -> None:
        """Initialize Sequential Quadratic Programming.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            constraints: Inequality constraints g(x) <= 0. Defaults to None.
            eq_constraints: Equality constraints h(x) = 0. Defaults to None.
            max_iter: Maximum iterations. Defaults to 100.
            tol: Convergence tolerance. Defaults to 1e-6.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.constraints = constraints or []
        self.eq_constraints = eq_constraints or []
        self.tol = tol

    def _numerical_gradient(
        self, func: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-7
    ) -> np.ndarray:
        """Compute numerical gradient using central differences.

        Args:
            func: Function to differentiate.
            x: Point at which to compute gradient.
            eps: Finite difference step size.

        Returns:
            Gradient vector.
        """
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
        return grad

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Sequential Quadratic Programming algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Build scipy constraint dictionaries
        scipy_constraints = []

        for g in self.constraints:
            scipy_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, g=g: -g(x),  # scipy uses g(x) >= 0
                }
            )

        for h in self.eq_constraints:
            scipy_constraints.append({"type": "eq", "fun": h})

        bounds = [(self.lower_bound, self.upper_bound)] * self.dim

        # Multi-start optimization
        best_solution = None
        best_fitness = np.inf

        n_starts = max(1, self.max_iter // 10)

        for _ in range(n_starts):
            # Random starting point
            x0 = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

            try:
                result = minimize(
                    self.func,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=scipy_constraints,
                    options={"maxiter": self.max_iter // n_starts, "ftol": self.tol},
                )

                if result.fun < best_fitness:
                    best_solution = result.x.copy()
                    best_fitness = result.fun

            except Exception:
                continue

        if best_solution is None:
            best_solution = np.random.uniform(
                self.lower_bound, self.upper_bound, self.dim
            )
            best_fitness = self.func(best_solution)

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import sphere

    # Equality constraint: sum(x) = 1
    def eq_constraint(x: np.ndarray) -> float:
        return np.sum(x) - 1

    # Inequality constraint: x[0] >= 0.2 (i.e., 0.2 - x[0] <= 0)
    def ineq_constraint(x: np.ndarray) -> float:
        return 0.2 - x[0]

    optimizer = SequentialQuadraticProgramming(
        func=sphere,
        lower_bound=-5,
        upper_bound=5,
        dim=2,
        constraints=[ineq_constraint],
        eq_constraints=[eq_constraint],
        max_iter=100,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
    print(f"Sum of x: {np.sum(best_solution):.6f} (target: 1)")
    print(f"x[0] >= 0.2: {best_solution[0] >= 0.2}")
