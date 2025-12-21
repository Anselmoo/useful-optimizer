"""AdaMax Optimizer.

This module implements the AdaMax optimization algorithm. AdaMax is a variant of Adam
that uses the infinity norm instead of the L2 norm for the second moment estimate.
This makes it less sensitive to outliers in gradients and can be more stable in some cases.

AdaMax performs the following update rule:
    m = beta1 * m + (1 - beta1) * gradient
    u = max(beta2 * u, |gradient|)
    x = x - (learning_rate / (1 - beta1^t)) * (m / u)

where:
    - x: current solution
    - m: first moment estimate (exponential moving average of gradients)
    - u: second moment estimate (exponential moving average of infinity norm of gradients)
    - learning_rate: step size for parameter updates
    - beta1, beta2: exponential decay rates for moment estimates
    - t: time step

Example:
    optimizer = AdaMax(func=objective_function, learning_rate=0.002, beta1=0.9, beta2=0.999,
                      lower_bound=-5, upper_bound=5, dim=2)
    best_solution, best_fitness = optimizer.search()

Attributes:
    func (Callable): The objective function to optimize.
    learning_rate (float): The learning rate for the optimization.
    beta1 (float): Exponential decay rate for first moment estimates.
    beta2 (float): Exponential decay rate for second moment estimates.
    epsilon (float): Small constant for numerical stability.

Methods:
    search(): Perform the AdaMax optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import approx_fprime

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class AdaMax(AbstractOptimizer):
    """AdaMax optimizer implementation.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        learning_rate (float, optional): The learning rate. Defaults to 0.002.
        beta1 (float, optional): Exponential decay rate for first moment estimates. Defaults to 0.9.
        beta2 (float, optional): Exponential decay rate for second moment estimates. Defaults to 0.999.
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-8.
        seed (int | None, optional): The seed value for random number generation. Defaults to None.


    Example:
        >>> from opt.gradient_based.adamax import AdaMax
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = AdaMax(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = AdaMax(
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
        learning_rate: float = 0.002,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        seed: int | None = None,
    ) -> None:
        """Initialize the AdaMax optimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the AdaMax optimization search.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        # Initialize solution randomly
        best_solution = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        best_fitness = self.func(best_solution)

        current_solution = best_solution.copy()
        m = np.zeros(self.dim)  # First moment estimate
        u = np.zeros(self.dim)  # Infinity norm-based second moment estimate

        for t in range(1, self.max_iter + 1):
            # Compute gradient at current position
            gradient = self._compute_gradient(current_solution)

            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * gradient

            # Update the exponentially weighted infinity norm
            u = np.maximum(self.beta2 * u, np.abs(gradient))

            # Compute bias-corrected first moment estimate
            bias_correction = 1 - np.power(self.beta1, t)

            # Update solution using AdaMax rule
            current_solution = current_solution - (
                self.learning_rate / bias_correction
            ) * (m / (u + self.epsilon))

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

    run_demo(AdaMax)
