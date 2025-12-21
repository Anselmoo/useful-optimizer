"""Firefly Algorithm implementation.

This module provides an implementation of the Firefly Algorithm optimization algorithm.
The Firefly Algorithm is a metaheuristic optimization algorithm inspired by the
flashing behavior of fireflies. It is commonly used to solve optimization problems by
simulating the behavior of fireflies in attracting each other.

The algorithm works by representing potential solutions as fireflies in a search space.
Each firefly's brightness is determined by its fitness value, with brighter fireflies
representing better solutions. Fireflies move towards brighter fireflies in the search
space, and their movements are influenced by attractiveness and light absorption
coefficients.

This implementation provides a class called FireflyAlgorithm, which can be used to
perform optimization using the Firefly Algorithm. The class takes an objective
function, lower and upper bounds of the search space, dimensionality of the search
space, and other optional parameters. The search method of the class runs the
Firefly Algorithm optimization and returns the best solution found.

Example usage:
    optimizer = FireflyAlgorithm(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=32.768,
        population_size=100,
        max_iter=1000,
        alpha=0.5,
        beta_0=1,
        gamma=1,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class FireflyAlgorithm(AbstractOptimizer):
    """Implementation of the Firefly Algorithm optimization algorithm.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The number of fireflies in the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        alpha (float, optional): The attractiveness coefficient. Defaults to 0.5.
        beta_0 (float, optional): The initial value of the light absorption coefficient. Defaults to 1.
        gamma (float, optional): The light absorption coefficient decay rate. Defaults to 1.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Attributes:
        alpha (float): The attractiveness coefficient.
        beta_0 (float): The initial value of the light absorption coefficient.
        gamma (float): The light absorption coefficient decay rate.

    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 100,
        max_iter: int = 1000,
        alpha: float = 0.5,
        beta_0: float = 1,
        gamma: float = 1,
        seed: int | None = None,
    ) -> None:
        """Initialize the FireflyAlgorithm class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.alpha = alpha
        self.beta_0 = beta_0
        self.gamma = gamma

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Firefly Algorithm optimization.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.

        """
        # Initialize population
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(self.func, 1, population)

        for _ in range(self.max_iter):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[i] < fitness[j]:
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta_0 * np.exp(-self.gamma * r**2)
                        population[j] += beta * (
                            population[i] - population[j]
                        ) + self.alpha * np.random.default_rng(self.seed).uniform(
                            -1, 1, self.dim
                        )
                        population[j] = np.clip(
                            population[j], self.lower_bound, self.upper_bound
                        )
                        fitness[j] = self.func(population[j])

        best_index = fitness.argmin()
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        return best_solution, best_fitness


if __name__ == "__main__":
    optimizer = FireflyAlgorithm(
        func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
