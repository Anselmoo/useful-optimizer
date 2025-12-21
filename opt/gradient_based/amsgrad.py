"""AMSGrad Optimizer.

This module implements the AMSGrad optimization algorithm. AMSGrad is a variant of Adam
that fixes the exponential moving average issue in Adam. It ensures that the second moment
estimate never decreases, which helps with convergence to the optimal solution.

AMSGrad performs the following update rule:
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient^2
    v_hat = max(v_hat, v)
    m_hat = m / (1 - beta1^t)
    v_hat_corrected = v_hat / (1 - beta2^t)
    x = x - learning_rate * m_hat / (sqrt(v_hat_corrected) + epsilon)

where:
    - x: current solution
    - m: first moment estimate (exponential moving average of gradients)
    - v: second moment estimate (exponential moving average of squared gradients)
    - v_hat: maximum of all v up to current time step
    - learning_rate: step size for parameter updates
    - beta1, beta2: exponential decay rates for moment estimates
    - epsilon: small constant for numerical stability
    - t: time step

Example:
    optimizer = AMSGrad(func=objective_function, learning_rate=0.001, beta1=0.9, beta2=0.999,
                       lower_bound=-5, upper_bound=5, dim=2)
    best_solution, best_fitness = optimizer.search()

Attributes:
    func (Callable): The objective function to optimize.
    learning_rate (float): The learning rate for the optimization.
    beta1 (float): Exponential decay rate for first moment estimates.
    beta2 (float): Exponential decay rate for second moment estimates.
    epsilon (float): Small constant for numerical stability.

Methods:
    search(): Perform the AMSGrad optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import approx_fprime

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class AMSGrad(AbstractOptimizer):
    """AMSGrad optimizer implementation.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        learning_rate (float, optional): The learning rate. Defaults to 0.001.
        beta1 (float, optional): Exponential decay rate for first moment estimates. Defaults to 0.9.
        beta2 (float, optional): Exponential decay rate for second moment estimates. Defaults to 0.999.
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-8.
        seed (int | None, optional): The seed value for random number generation. Defaults to None.


    Example:
        >>> from opt.gradient_based.amsgrad import AMSGrad
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = AMSGrad(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = AMSGrad(
        ...     func=shifted_ackley, dim=2,
        ...     lower_bound=-2.768, upper_bound=2.768,
        ...     max_iter=10, seed=42
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
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        seed: int | None = None,
    ) -> None:
        """Initialize the AMSGrad optimizer."""
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
        """Perform the AMSGrad optimization search.

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
        v = np.zeros(self.dim)  # Second moment estimate
        v_hat = np.zeros(self.dim)  # Maximum of second moment estimates

        for t in range(1, self.max_iter + 1):
            # Compute gradient at current position
            gradient = self._compute_gradient(current_solution)

            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * gradient

            # Update biased second moment estimate
            v = self.beta2 * v + (1 - self.beta2) * np.square(gradient)

            # Update the maximum of second moment estimates (key difference from Adam)
            v_hat = np.maximum(v_hat, v)

            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - np.power(self.beta1, t))

            # Compute bias-corrected second moment estimate using v_hat
            v_hat_corrected = v_hat / (1 - np.power(self.beta2, t))

            # Update solution using AMSGrad rule
            current_solution = current_solution - self.learning_rate * m_hat / (
                np.sqrt(v_hat_corrected) + self.epsilon
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
    from opt.demo import run_demo

    run_demo(AMSGrad)
