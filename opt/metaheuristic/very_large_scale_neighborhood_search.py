"""Very Large Scale Neighborhood Search (VLSN) Algorithm.

This module implements the Very Large Scale Neighborhood Search (VLSN) optimization
algorithm. VLSN is a local search method used for mathematical optimization.
It explores very large neighborhoods with an efficient algorithm.

The main idea behind VLSN is to perform a search in a large-scale neighborhood to find
the optimal solution for a given function. The size of the neighborhood is defined by
the `neighborhood_size` parameter.
The larger the neighborhood size, the more potential solutions the algorithm will
consider at each step, but the more computational resources it will require.

VLSN is particularly useful for problems where the search space is large and complex,
and where traditional optimization methods may not be applicable.

Example:
    optimizer = VeryLargeScaleNeighborhood(
        func=objective_function,
        lower_bound=-10,
        upper_bound=10,
        dim=2,
        population_size=100,
        max_iter=1000,
        neighborhood_size=10
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    func (Callable): The objective function to optimize.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimension of the search space.
    population_size (int, optional): The size of the population. Defaults to 100.
    max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
    neighborhood_size (int, optional): The size of the neighborhood. Defaults to 10.
    seed (Optional[int], optional): The seed for the random number generator. Defaults to None.

Methods:
    search(): Perform the VLSN optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class VeryLargeScaleNeighborhood(AbstractOptimizer):
    """A class representing the Very Large Scale Neighborhood optimizer.

    This optimizer performs a search in a large-scale neighborhood to find the optimal solution
    for a given function.

    Parameters:
    - func (Callable[[ndarray], float]): The objective function to be optimized.
    - lower_bound (float): The lower bound of the search space.
    - upper_bound (float): The upper bound of the search space.
    - dim (int): The dimensionality of the search space.
    - population_size (int, optional): The size of the population. Defaults to 100.
    - max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
    - neighborhood_size (int, optional): The size of the neighborhood. Defaults to 10.
    - seed (Optional[int], optional): The seed for the random number generator. Defaults to None.


    Example:
        >>> from opt.metaheuristic.very_large_scale_neighborhood_search import (
        ...     VeryLargeScaleNeighborhood,
        ... )
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = VeryLargeScaleNeighborhood(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = VeryLargeScaleNeighborhood(
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
        population_size: int = 100,
        max_iter: int = 1000,
        neighborhood_size: int = 10,
        seed: int | None = None,
    ) -> None:
        """Initialize the Very Large Scale Neighborhood optimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.neighborhood_size = neighborhood_size
        self.population: np.ndarray = np.empty((self.population_size, self.dim))

    def initialize_population(self) -> None:
        """Initializes the population by generating random individuals within the search space."""
        self.population = self.lower_bound + np.random.default_rng(self.seed).uniform(
            size=(self.population_size, self.dim)
        ) * (self.upper_bound - self.lower_bound)

    def search(self) -> tuple[np.ndarray, float]:
        """Performs the search using the Very Large Scale Neighborhood algorithm.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best individual found and its fitness value.
        """
        self.initialize_population()
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                best_neighbor = None
                best_fitness = np.inf
                for _ in range(self.neighborhood_size):
                    neighbor = self.population[i] + np.random.default_rng(
                        self.seed
                    ).uniform(-1, 1, self.dim)
                    neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
                    fitness = self.func(neighbor)
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_neighbor = neighbor
                self.population[i] = best_neighbor
        best_index = np.argmin(
            [self.func(individual) for individual in self.population]
        )
        return self.population[best_index], self.func(self.population[best_index])


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(VeryLargeScaleNeighborhood)
