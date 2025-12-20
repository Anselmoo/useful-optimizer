"""AdaDelta Optimizer.

This module implements the AdaDelta optimizer, which is an extension of AdaGrad that
seeks to reduce its sensitivity to the learning rate hyperparameter.

AdaDelta is a gradient-based optimization algorithm that adapts the learning rate
for each of the parameters in the model. It is designed to converge faster than
AdaGrad by using a moving average of the squared gradient values to scale the learning rate.

The AdaDelta optimizer is defined by the following update rule:

    Eg = rho * Eg + (1 - rho) * g^2
    dx = -sqrt(Edx + eps) / sqrt(Eg + eps) * g
    Edx = rho * Edx + (1 - rho) * dx^2
    x = x + dx

where:
    - x: current solution
    - g: gradient of the objective function
    - rho: decay rate
    - eps: small constant to avoid dividing by zero
    - Eg: moving average of squared gradient values
    - Edx: moving average of squared updates

The algorithm iteratively updates the solution x by computing the gradient of the
objective function at x, scaling it by the moving average of the squared gradients,
and dividing it by the square root of the moving average of the squared updates.

The algorithm continues for a fixed number of iterations or until a specified
stopping criterion is met, returning the best solution found.

This module provides a simple example of how to use the AdaDelta optimizer to minimize
the Shifted Ackley's function in two dimensions.

Example:
    $ python adadelta.py

Attributes:
    rho (float): Decay rate for the moving average of squared gradients
    eps (float): Small constant to avoid dividing by zero
    Eg (np.ndarray): Moving average of squared gradient values
    Edx (np.ndarray): Moving average of squared updates
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


class AdaDelta(AbstractOptimizer):
    """AdaDelta optimizer implementation.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        seed (Optional[int], optional): The seed value for random number generation. Defaults to None.
        rho (float, optional): The decay rate for the moving average of squared gradients. Defaults to 0.97.
        eps (float, optional): A small constant to avoid division by zero. Defaults to 1e-8.
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        rho: float = 0.97,
        eps: float = 1e-8,
        seed: int | None = None,
    ) -> None:
        """Initialize the AdaDelta optimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.rho = rho
        self.eps = eps
        self.Eg = np.zeros(self.dim)
        self.Edx = np.zeros(self.dim)

    def _update(self, x: np.ndarray) -> np.ndarray:
        """Perform a single update step of the AdaDelta algorithm.

        Args:
            x (np.ndarray): The current solution.

        Returns:
            np.ndarray: The updated solution.
        """
        g = approx_fprime(x, self.func, self.eps)
        self.Eg = self.rho * self.Eg + (1 - self.rho) * g**2
        dx = -np.sqrt(self.Edx + self.eps) / np.sqrt(self.Eg + self.eps) * g
        self.Edx = self.rho * self.Edx + (1 - self.rho) * dx**2
        return x + dx

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the AdaDelta search.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        # Initialize solution
        x = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        best_solution = x
        best_fitness = np.inf

        # Main loop
        for _ in range(self.max_iter):
            x = self._update(x)
            x = np.clip(x, self.lower_bound, self.upper_bound)
            fitness = self.func(x)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = x

        return best_solution, best_fitness


if __name__ == "__main__":
    optimizer = AdaDelta(
        func=shifted_ackley, lower_bound=-2.768, upper_bound=+2.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
