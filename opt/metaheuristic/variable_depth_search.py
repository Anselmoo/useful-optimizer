"""Variable Depth Search (VDS) Algorithm.

This module implements the Variable Depth Search (VDS) optimization algorithm. VDS is a
local search method used for mathematical optimization. It explores the search space by
variable-depth first search and backtracking.

The main idea behind VDS is to perform a search in a variable depth to find the optimal
solution for a given function. The depth of the search is defined by the `depth`
parameter. The larger the depth, the more potential solutions the algorithm will
consider at each step, but the more computational resources it will require.

VDS is particularly useful for problems where the search space is large and complex, and
where traditional optimization methods may not be applicable.

Example:
    optimizer = VariableDepthSearch(
        func=objective_function,
        lower_bound=-10,
        upper_bound=10,
        dim=2,
        population_size=100,
        max_iter=1000,
        depth=10
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
    depth (int, optional): The depth of the search. Defaults to 10.
    seed (Optional[int], optional): The seed for the random number generator. Defaults to None.

Methods:
    search(): Perform the VDS optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class VariableDepthSearch(AbstractOptimizer):
    """Implementation of the Variable Depth Search optimizer.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        max_depth (int, optional): The maximum depth of the search. Defaults to 20.
        seed (Optional[int], optional): The seed value for random number generation. Defaults to None.


    Example:
        >>> from opt.metaheuristic.variable_depth_search import VariableDepthSearch
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = VariableDepthSearch(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = VariableDepthSearch(
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
        max_depth: int = 20,
        seed: int | None = None,
    ) -> None:
        """Initialize the Variable Depth Search optimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.max_depth = max_depth
        self.population: np.ndarray = np.empty((self.population_size, self.dim))

    def initialize_population(self) -> None:
        """Initialize the population by generating random individuals within the search space."""
        self.population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Variable Depth Search algorithm.

        Returns:
            Tuple[np.ndarray, float]: The best solution found and its corresponding fitness value.
        """
        self.initialize_population()
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                best_solution = self.population[i]
                best_fitness = self.func(best_solution)
                for depth in range(1, self.max_depth + 1):
                    new_solution = best_solution + np.random.default_rng(
                        self.seed
                    ).uniform(-depth, depth, size=self.dim)
                    new_solution = np.clip(
                        new_solution, self.lower_bound, self.upper_bound
                    )
                    new_fitness = self.func(new_solution)
                    if new_fitness < best_fitness:
                        best_solution = new_solution
                        best_fitness = new_fitness
                self.population[i] = best_solution
        best_index = np.argmin(
            [self.func(individual) for individual in self.population]
        )
        return self.population[best_index], self.func(self.population[best_index])


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(VariableDepthSearch)
