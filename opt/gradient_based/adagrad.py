"""ADAGrad Optimizer.

This module implements the Adaptive Gradient Algorithm (ADAGrad) optimizer. ADAGrad is
a gradient-based optimization algorithm that adapts the learning rate to the parameters,
performing smaller updates for parameters associated with frequently occurring features,
and larger updates for parameters associated with infrequent features. It is particularly
useful for dealing with sparse data.

ADAGrad's main strength is that it eliminates the need to manually tune the learning rate.
Most implementations also include a 'smoothing term' to avoid division by zero when the
gradient is zero.

The ADAGrad optimizer is commonly used in machine learning and deep learning applications.

Example:
    optimizer = ADAGrad(func=objective_function, learning_rate=0.01, initial_guess=[0, 0])
    best_solution, best_fitness = optimizer.optimize()

Attributes:
    func (Callable): The objective function to optimize.
    learning_rate (float): The learning rate for the optimization.
    initial_guess (List[float]): The starting point for the optimization.

Methods:
    optimize(): Perform the ADAGrad optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import approx_fprime

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ADAGrad(AbstractOptimizer):
    """ADAGrad optimizer implementation.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        seed (Optional[int], optional): The seed value for random number generation. Defaults to None.
        lr (float, optional): The learning rate. Defaults to 0.01.
        eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-8.


    Example:
        >>> from opt.gradient_based.adagrad import ADAGrad
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ADAGrad(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ADAGrad(
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
        lr: float = 0.01,
        eps: float = 1e-8,
        seed: int | None = None,
    ) -> None:
        """Initialize the ADAGrad optimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.lr = lr
        self.eps = eps

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the ADAGrad search algorithm.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        x = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        grad_accumulator = np.zeros(self.dim)

        for _ in range(self.max_iter):
            grad = approx_fprime(x, self.func, np.sqrt(np.finfo(float).eps))
            grad_accumulator += grad**2
            adjusted_grad = grad / (np.sqrt(grad_accumulator) + self.eps)
            x = x - self.lr * adjusted_grad

        best_solution = x
        best_fitness = self.func(best_solution)
        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ADAGrad)
