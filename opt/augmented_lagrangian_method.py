"""Augmented Lagrangian optimizer.

This module implements an optimizer based on the Augmented Lagrangian method. The
Augmented Lagrangian method is an optimization technique that combines the
advantages of both penalty and Lagrange multiplier methods. It is commonly used to
solve constrained optimization problems.

The `AugmentedLagrangian` class is the main class of this module. It takes an objective
function, lower and upper bounds of the search space, dimensionality of the search
space, and other optional parameters as input. It performs the search using the
Augmented Lagrangian method and returns the best solution found and its fitness value.

Example usage:
    optimizer = AugmentedLagrangian(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=+2.768,
        dim=2,
        max_iter=8000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

Note:
    This module requires the `scipy` library to be installed.

"""

# Rest of the code...

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import minimize

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class AugmentedLagrangian(AbstractOptimizer):
    """Augmented Lagrangian optimizer.

    Args:
        func: The objective function to be minimized.
        lower_bound: The lower bound of the search space.
        upper_bound: The upper bound of the search space.
        dim: The dimensionality of the search space.
        max_iter: The maximum number of iterations.
        c: The penalty parameter for the constraint violation.
        lambda_: The Lagrange multiplier.
        static_cost: The cost assigned to infeasible solutions.
        seed: The random seed for reproducibility.

    Attributes:
        solution: The best solution found by the optimizer.
        c: The penalty parameter for the constraint violation.
        lambda_: The Lagrange multiplier.
        static_cost: The cost assigned to infeasible solutions.

    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        c: float = 1,
        lambda_: float = 0.1,
        static_cost: float = 1e10,
        seed: int | None = None,
    ) -> None:
        """Initialize the AugmentedLagrangian class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.solution: np.ndarray | None = None
        self.c: float = c
        self.lambda_: float = lambda_
        self.static_cost: float = static_cost

    def constraint(self, x: np.ndarray) -> float:
        """Constraint function.

        Args:
            x: The input vector.

        Returns:
            The value of the constraint function.

        """
        return np.sum(x) - 1

    def augmented_lagrangian_func(
        self, x: np.ndarray, lambda_: float, c: float
    ) -> float:
        """Augmented Lagrangian function.

        Args:
            x: The input vector.
            lambda_: The Lagrange multiplier.
            c: The penalty parameter for the constraint violation.

        Returns:
            The value of the augmented Lagrangian function.

        """
        constraint_val = self.constraint(x)
        if constraint_val < 0:
            constraint_val = 0
        if np.isnan(constraint_val):
            constraint_val = self.static_cost

        return (
            self.func(x)
            - lambda_ * constraint_val
            + (c / 2) * np.square(constraint_val)
        )

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the search.

        Returns:
            A tuple containing the best solution found and its fitness value.

        """
        best_solution: np.ndarray = np.empty(self.dim)
        best_fitness: float = np.inf

        x0 = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        for _ in range(self.max_iter):
            res = minimize(
                self.augmented_lagrangian_func,
                x0,
                args=(self.lambda_, self.c),
                method="L-BFGS-B",
                bounds=[(self.lower_bound, self.upper_bound)] * self.dim,
            )
            if res is None or not res.success:
                continue
            if self.constraint(res.x) > 0:
                self.lambda_ = self.lambda_ - self.c * self.constraint(res.x)
            self.c = self.c * (1.1 if self.constraint(res.x) < 0 else 1.5)
            x0 = res.x
            if res.fun < best_fitness:
                best_solution = res.x
                best_fitness = res.fun

        self.solution = best_solution
        return self.solution, self.func(self.solution)


if __name__ == "__main__":
    optimizer = AugmentedLagrangian(
        func=shifted_ackley, lower_bound=-2.768, upper_bound=+2.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
