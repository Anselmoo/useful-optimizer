"""Nesterov Accelerated Gradient (NAG) Optimizer.

This module implements the Nesterov Accelerated Gradient optimization algorithm. NAG is
an improvement over SGD with Momentum that provides better convergence rates. The key
idea is to compute the gradient not at the current position, but at an approximate
future position, which provides better gradient information.

NAG performs the following update rule:
    v = momentum * v - learning_rate * gradient(x + momentum * v)
    x = x + v

where:
    - x: current solution
    - v: velocity (momentum term)
    - learning_rate: step size for parameter updates
    - momentum: momentum coefficient (typically 0.9)
    - gradient: gradient of the objective function

Example:
    optimizer = NesterovAcceleratedGradient(func=objective_function, learning_rate=0.01,
                                          momentum=0.9, lower_bound=-5, upper_bound=5, dim=2)
    best_solution, best_fitness = optimizer.search()

Attributes:
    func (Callable): The objective function to optimize.
    learning_rate (float): The learning rate for the optimization.
    momentum (float): The momentum coefficient.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimensionality of the search space.

Methods:
    search(): Perform the NAG optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import approx_fprime

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class NesterovAcceleratedGradient(AbstractOptimizer):
    """Nesterov Accelerated Gradient optimizer implementation.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        momentum (float, optional): The momentum coefficient. Defaults to 0.9.
        seed (int | None, optional): The seed value for random number generation. Defaults to None.


    Example:
        >>> from opt.gradient_based.nesterov_accelerated_gradient import NesterovAcceleratedGradient
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = NesterovAcceleratedGradient(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = NesterovAcceleratedGradient(
        ...     func=shifted_ackley,
        ...     dim=2,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     max_iter=10,
        ...     seed=42,
        ... )
        >>> _, fitness = optimizer.search()
        >>> isinstance(float(fitness), float)
        True
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        seed: int | None = None,
    ) -> None:
        """Initialize the Nesterov Accelerated Gradient optimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.learning_rate = learning_rate
        self.momentum = momentum

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the Nesterov Accelerated Gradient optimization search.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        # Initialize solution randomly
        best_solution = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        best_fitness = self.func(best_solution)

        current_solution = best_solution.copy()
        velocity = np.zeros(self.dim)  # Initialize velocity to zero

        for _ in range(self.max_iter):
            # Compute the lookahead position
            lookahead_position = current_solution + self.momentum * velocity

            # Apply bounds to lookahead position
            lookahead_position = np.clip(
                lookahead_position, self.lower_bound, self.upper_bound
            )

            # Compute gradient at lookahead position (this is the key difference from SGD with momentum)
            gradient = self._compute_gradient(lookahead_position)

            # Update velocity using momentum and gradient at lookahead position
            velocity = self.momentum * velocity - self.learning_rate * gradient

            # Update solution using velocity
            current_solution = current_solution + velocity

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
    from opt.demo import run_demo

    run_demo(NesterovAcceleratedGradient)
