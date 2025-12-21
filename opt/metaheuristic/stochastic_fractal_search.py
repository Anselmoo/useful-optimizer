"""Stochastic Diffusion Search optimizer.

This module implements the Stochastic Fractal Search optimizer, which is an
optimization algorithm used to find the minimum of a given function.

The Stochastic Fractal Search algorithm works by maintaining a population of
individuals and iteratively updating them based on their scores. At each iteration,
a best individual is selected, and other individuals in the population undergo a
diffusion phase to explore the search space. The algorithm continues for a specified
number of iterations or until a termination condition is met.

Example:
    To use the Stochastic Fractal Search optimizer, create an instance of the
    `StochasticFractalSearch` class and call the `search` method:

    ```python
    optimizer = StochasticFractalSearch(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    ```

    This will return the best solution found and its corresponding fitness value.

Attributes:
    diffusion_parameter (float): The diffusion parameter used in the diffusion phase of the algorithm.
    population (np.ndarray): The population of individuals.
    scores (np.ndarray): The scores of the individuals in the population.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class StochasticFractalSearch(AbstractOptimizer):
    """Stochastic Fractal Search optimizer.

    This optimizer uses the Stochastic Fractal Search algorithm to find the minimum of a given function.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        diffusion_parameter (float, optional): The diffusion parameter. Defaults to 0.5.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Attributes:
        diffusion_parameter (float): The diffusion parameter.
        population (np.ndarray): The population of individuals.
        scores (np.ndarray): The scores of the individuals in the population.

    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 100,
        max_iter: int = 1000,
        diffusion_parameter: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize the StochasticFractalSearch class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.diffusion_parameter = diffusion_parameter
        self.population: np.ndarray = np.empty((self.population_size, self.dim))
        self.scores = np.empty(self.population_size)

    def initialize_population(self) -> None:
        """Initialize the population of individuals.

        This method initializes the population of individuals by randomly sampling from the search space.

        """
        self.population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        self.scores = np.array(
            [self.func(individual) for individual in self.population]
        )

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the stochastic fractal search.

        This method performs the stochastic fractal search algorithm to find the minimum of the objective function.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best individual found and its corresponding score.

        """
        self.initialize_population()
        for _ in range(self.max_iter):
            self.seed += 1
            best_index = np.argmin(self.scores)
            for i in range(self.population_size):
                if np.random.default_rng(self.seed).random() < self.fractal_dimension(
                    self.population[i]
                ):
                    self.seed += 1
                    self.population[i] = self.diffusion_phase(
                        self.population[best_index]
                    )
                    self.population[i] = np.clip(
                        self.population[i], self.lower_bound, self.upper_bound
                    )
                    self.scores[i] = self.func(self.population[i])
        best_index = np.argmin(self.scores)
        return self.population[best_index], self.scores[best_index]

    def fractal_dimension(self, x: np.ndarray) -> float:
        """Calculate the fractal dimension.

        This method calculates the fractal dimension of an individual.

        Args:
            x (np.ndarray): The individual to calculate the fractal dimension for.

        Returns:
            float: The fractal dimension of the individual.
        """
        return np.sum(np.abs(x - self.population.mean(axis=0))) / (
            self.dim * self.population.std()
        )

    def diffusion_phase(self, x: np.ndarray) -> np.ndarray:
        """Perform the diffusion phase.

        This method performs the diffusion phase of the algorithm.

        Args:
            x (np.ndarray): The individual to perform the diffusion phase on.

        Returns:
            np.ndarray: The individual after the diffusion phase.
        """
        return x + self.diffusion_parameter * np.random.default_rng(self.seed).uniform(
            -1, 1, self.dim
        )


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(StochasticFractalSearch)
