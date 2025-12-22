"""Adaptive Moment Estimation (Adam) Optimizer.

This module implements the Adam optimization algorithm. Adam is a gradient-based
optimization algorithm that computes adaptive learning rates for each parameter. It
combines the advantages of two other extensions of stochastic gradient descent:

    - AdaGrad
    - RMSProp

Adam works well in practice and compares favorably to other adaptive learning-method
algorithms as it converges fast and the learning speed of the Model is quite fast and
efficient. It is straightforward to implement, is computationally efficient, has little
memory requirements, is invariant to diagonal rescaling of the gradients, and is well
suited for problems that are large in terms of data and/or parameters.

Example:
    optimizer = Adam(func=objective_function, learning_rate=0.01, initial_guess=[0, 0])
    best_solution, best_fitness = optimizer.optimize()

Attributes:
    func (Callable): The objective function to optimize.
    learning_rate (float): The learning rate for the optimization.
    initial_guess (List[float]): The starting point for the optimization.

Methods:
    optimize(): Perform the Adam optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import approx_fprime

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ADAMOptimization(AbstractOptimizer):
    """ADAMOptimization is a class that implements the ADAM optimization algorithm.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        alpha (float, optional): The learning rate. Defaults to 0.001.
        beta1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
        epsilon (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-13.
        seed (Optional[int], optional): The random seed for reproducibility. Defaults to None.


    Example:
        >>> from opt.gradient_based.adaptive_moment_estimation import ADAMOptimization
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ADAMOptimization(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ADAMOptimization(
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
        alpha: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-13,
        seed: int | None = None,
    ) -> None:
        """Initialize the ADAM optimization algorithm."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the ADAM optimization search.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        m = np.zeros(self.dim)
        v = np.zeros(self.dim)
        best_solution = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        best_fitness = self.func(best_solution)

        for t in range(1, self.max_iter + 1):
            grad = self._compute_gradient(best_solution)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * np.square(grad)
            m_hat = m / (1 - np.power(self.beta1, t))
            v_hat = v / (1 - np.power(self.beta2, t))

            best_solution = best_solution - self.alpha * m_hat / (
                np.sqrt(v_hat) + self.epsilon
            )
            best_solution = np.clip(best_solution, self.lower_bound, self.upper_bound)

            fitness = self.func(best_solution)
            best_fitness = min(best_fitness, fitness)

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

    run_demo(ADAMOptimization)
