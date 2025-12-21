"""Cat Swarm Optimization (CSO) algorithm.

This module implements the Cat Swarm Optimization (CSO) algorithm, which is a
population-based optimization algorithm inspired by the behavior of cats. The algorithm
aims to find the optimal solution for a given optimization problem by simulating the
hunting behavior of cats.

The CSO algorithm is implemented in the `CatSwarmOptimization` class, which inherits
from the `AbstractOptimizer` class. The `CatSwarmOptimization` class provides methods
to initialize the population, perform seeking mode and tracing mode operations, and run
the CSO algorithm to find the optimal solution.

Example usage:
    optimizer = CatSwarmOptimization(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        cats=100,
        max_iter=2000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness: {best_fitness}")

Attributes:
    seeking_memory_pool (int): The size of the seeking memory pool.
    counts_of_dimension_to_change (int): The number of dimensions to change during seeking mode.
    smp_change_probability (float): The probability of changing dimensions during seeking mode.
    spc_probability (float): The probability of performing tracing mode.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class CatSwarmOptimization(AbstractOptimizer):
    """Cat Swarm Optimization (CSO) algorithm.

    Cat Swarm Optimization (CSO) is a population-based optimization algorithm inspired by the behavior of cats.
    It aims to find the optimal solution for a given optimization problem by simulating the hunting behavior of cats.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        dim (int): The dimensionality of the problem.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        cats (int, optional): The number of cats in the population. Defaults to 50.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        seeking_memory_pool (int, optional): The size of the seeking memory pool. Defaults to 5.
        counts_of_dimension_to_change (int | None, optional): The number of dimensions to change during seeking mode.
            If None, it is set to `dim - 1`. Defaults to None.
        smp_change_probability (float, optional): The probability of changing dimensions during seeking mode.
            Defaults to 0.1.
        spc_probability (float, optional): The probability of performing tracing mode. Defaults to 0.2.
        seed (int | None, optional): The seed value for random number generation. Defaults to None.

    Attributes:
        seeking_memory_pool (int): The size of the seeking memory pool.
        counts_of_dimension_to_change (int): The number of dimensions to change during seeking mode.
        smp_change_probability (float): The probability of changing dimensions during seeking mode.
        spc_probability (float): The probability of performing tracing mode.

    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        dim: int,
        lower_bound: float,
        upper_bound: float,
        cats: int = 50,
        max_iter: int = 1000,
        seeking_memory_pool: int = 5,
        counts_of_dimension_to_change: int | None = None,
        smp_change_probability: float = 0.1,
        spc_probability: float = 0.2,
        seed: int | None = None,
    ) -> None:
        """Initialize the CatSwarmOptimization class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=cats,
        )
        self.seeking_memory_pool = seeking_memory_pool
        if counts_of_dimension_to_change is None:
            counts_of_dimension_to_change = dim - 1
        self.counts_of_dimension_to_change = counts_of_dimension_to_change
        self.smp_change_probability = smp_change_probability
        self.spc_probability = spc_probability

    def _initialize(self) -> np.ndarray:
        """Initialize the population by generating random solutions within the search space.

        Returns:
            np.ndarray: The initial population of cats.

        """
        return np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

    def _seeking_mode(self, population: np.ndarray) -> np.ndarray:
        """Perform the seeking mode operation on the population.

        Args:
            population (np.ndarray): The current population of cats.

        Returns:
            np.ndarray: The updated population after performing seeking mode.

        """
        new_population = []
        for cat in population:
            self.seed += 1
            if np.random.default_rng(self.seed).random() < self.smp_change_probability:
                cat[
                    np.random.default_rng(self.seed + 1).choice(
                        self.dim, self.counts_of_dimension_to_change, replace=False
                    )
                ] = np.random.default_rng(self.seed).uniform(
                    self.lower_bound, self.upper_bound
                )

            new_population.append(cat)
        return np.array(new_population)

    def _tracing_mode(self, population: np.ndarray, best_cat: np.ndarray) -> np.ndarray:
        """Perform the tracing mode operation on the population.

        Args:
            population (np.ndarray): The current population of cats.
            best_cat (np.ndarray): The best cat found so far.

        Returns:
            np.ndarray: The updated population after performing tracing mode.

        """
        return population + self.spc_probability * (best_cat - population)

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Cat Swarm Optimization algorithm to find the optimal solution.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best cat found and its corresponding fitness value.

        """
        population = self._initialize()
        best_cat: np.ndarray = np.array([])
        best_fitness = np.inf
        for _ in range(self.max_iter):
            self.seed += 1
            fitness = np.apply_along_axis(self.func, 1, population)
            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_cat = population[np.argmin(fitness)]
            if np.random.default_rng(self.seed).random() < self.spc_probability:
                population = self._tracing_mode(population, best_cat)
            else:
                population = self._seeking_mode(population)
        return best_cat, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(CatSwarmOptimization)
