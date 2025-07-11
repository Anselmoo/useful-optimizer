"""RMSprop Optimizer.

This module implements the RMSprop optimization algorithm. RMSprop is an adaptive
learning rate method that was proposed by Geoffrey Hinton. It modifies AdaGrad to
perform better in non-convex settings by using a moving average of squared gradients
instead of accumulating all squared gradients.

RMSprop performs the following update rule:
    v = rho * v + (1 - rho) * gradient^2
    x = x - (learning_rate / sqrt(v + epsilon)) * gradient

where:
    - x: current solution
    - v: moving average of squared gradients
    - learning_rate: step size for parameter updates
    - rho: decay rate (typically 0.9)
    - epsilon: small constant to avoid division by zero
    - gradient: gradient of the objective function at x

Example:
    optimizer = RMSprop(func=objective_function, learning_rate=0.01, rho=0.9,
                       lower_bound=-5, upper_bound=5, dim=2)
    best_solution, best_fitness = optimizer.search()

Attributes:
    func (Callable): The objective function to optimize.
    learning_rate (float): The learning rate for the optimization.
    rho (float): The decay rate for the moving average.
    epsilon (float): Small constant to avoid division by zero.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimensionality of the search space.

Methods:
    search(): Perform the RMSprop optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import approx_fprime

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class RMSprop(AbstractOptimizer):
    """RMSprop optimizer implementation.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        rho (float, optional): The decay rate for the moving average. Defaults to 0.9.
        epsilon (float, optional): Small constant to avoid division by zero. Defaults to 1e-8.
        seed (int | None, optional): The seed value for random number generation. Defaults to None.
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        rho: float = 0.9,
        epsilon: float = 1e-8,
        seed: int | None = None,
    ) -> None:
        """Initialize the RMSprop optimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the RMSprop optimization search.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        # Initialize solution randomly
        best_solution = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        best_fitness = self.func(best_solution)

        current_solution = best_solution.copy()
        v = np.zeros(self.dim)  # Initialize moving average of squared gradients

        for _ in range(self.max_iter):
            # Compute gradient at current position
            gradient = self._compute_gradient(current_solution)

            # Update moving average of squared gradients
            v = self.rho * v + (1 - self.rho) * np.square(gradient)

            # Update solution using RMSprop rule
            current_solution = (
                current_solution
                - (self.learning_rate / (np.sqrt(v) + self.epsilon)) * gradient
            )

            # Apply bounds
            current_solution = np.clip(
                current_solution, self.lower_bound, self.upper_bound
            )

            # Evaluate fitness
            current_fitness = self.func(current_solution)

            # Update best solution if improved
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness

        return best_solution, best_fitness

    def _compute_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function at a given point.

        Args:
            x (np.ndarray): The point at which to compute the gradient.

        Returns:
            np.ndarray: The gradient vector.
        """
        epsilon = np.sqrt(np.finfo(float).eps)
        return approx_fprime(x, self.func, epsilon)


if __name__ == "__main__":
    optimizer = RMSprop(
        func=shifted_ackley, lower_bound=-2.768, upper_bound=+2.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
