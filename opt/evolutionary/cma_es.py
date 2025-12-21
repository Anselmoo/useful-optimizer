"""Covariance Matrix Adaptation Evolution Strategy.

This module implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm,
which is a derivative-free optimization method that uses an evolutionary strategy to search for
the optimal solution. It adapts the covariance matrix of the multivariate Gaussian distribution
to guide the search towards promising regions of the search space.

The CMA-ES algorithm is implemented in the `CMAESAlgorithm` class, which inherits from the
`AbstractOptimizer` class. The `CMAESAlgorithm` class provides a `search` method that runs the
CMA-ES algorithm to search for the optimal solution.

Example usage:
    optimizer = CMAESAlgorithm(
        func=shifted_ackley,
        dim=2,
        lower_bound=-12.768,
        upper_bound=12.768,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.linalg import sqrtm

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class CMAESAlgorithm(AbstractOptimizer):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm.

    This algorithm is a derivative-free optimization method that uses an evolutionary strategy
    to search for the optimal solution. It adapts the covariance matrix of the multivariate
    Gaussian distribution to guide the search towards promising regions of the search space.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        dim (int): The dimensionality of the search space.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        population_size (int, optional): The number of solutions in each generation. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        sigma_init (float, optional): The initial step size. Defaults to 0.5.
        epsilon (float, optional): A small value to prevent the step size from becoming too small. Defaults to 1e-9.
        seed (Optional[int], optional): The random seed. Defaults to None.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the best solution found and its corresponding fitness value.


    Example:
        >>> from opt.evolutionary.cma_es import CMAESAlgorithm
        >>> from opt.benchmark.functions import rosenbrock
        >>> optimizer = CMAESAlgorithm(
        ...     func=rosenbrock, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = CMAESAlgorithm(
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
        dim: int,
        lower_bound: float,
        upper_bound: float,
        population_size: int = 100,
        max_iter: int = 1000,
        sigma_init: float = 0.5,
        epsilon: float = 1e-9,
        seed: int | None = None,
    ) -> None:
        """Initialize the CMAESAlgorithm class."""
        super().__init__(
            func=func,
            dim=dim,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            population_size=population_size,
            max_iter=max_iter,
            seed=seed,
        )
        self.sigma = sigma_init
        self.epsilon = epsilon

    def search(self) -> tuple[np.ndarray, float]:
        """Run the CMA-ES algorithm to search for the optimal solution.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best solution found and its corresponding fitness value.
        """
        # Initialize mean and covariance matrix
        mean = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        cov = np.eye(self.dim)

        # Initialize evolution paths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)

        # Other parameters
        mu = self.population_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        cc = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        cmu = min(
            1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff)
        )
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + cs

        h_sigma_threshold = 1.4
        for _ in range(self.max_iter):
            # Sample new solutions
            solutions = np.random.default_rng(self.seed + 1).multivariate_normal(
                mean, self.sigma**2 * cov, self.population_size
            )

            # Evaluate solutions
            fitness = np.apply_along_axis(self.func, 1, solutions)

            # Sort by fitness and compute weighted mean into center
            indices = np.argsort(fitness)
            mean_old = mean
            mean = np.dot(weights, solutions[indices[:mu]])

            # Update evolution paths
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * np.dot(
                np.linalg.inv(sqrtm(cov)), (mean - mean_old) / self.sigma
            )
            h_sigma = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - cs) ** (2 * (_ + 1)))
                / np.sqrt(self.dim)
                < h_sigma_threshold
            )
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (
                mean - mean_old
            ) / self.sigma

            # Adapt covariance matrix
            artmp = (1 / self.sigma) * (solutions[indices[:mu]] - mean_old)
            cov = (
                (1 - c1 - cmu) * cov
                + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cc * (2 - cc) * cov)
                + cmu * np.dot(artmp.T, np.dot(np.diag(weights), artmp))
            )

            # Adapt step size
            self.sigma *= np.exp(
                (cs / damps) * (np.linalg.norm(p_sigma) / np.sqrt(self.dim) - 1)
            )

            # Prevent sigma from becoming too small
            self.sigma = max(self.sigma, self.epsilon)

            # Adapt step size
            self.sigma *= np.exp(
                (cs / damps) * (np.linalg.norm(p_sigma) / np.sqrt(self.dim) - 1)
            )

        best_solution = mean
        best_fitness = self.func(best_solution)
        return best_solution, best_fitness


if __name__ == "__main__":
    optimizer = CMAESAlgorithm(
        func=shifted_ackley, dim=2, lower_bound=-12.768, upper_bound=12.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
