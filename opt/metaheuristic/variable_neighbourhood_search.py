"""Variable Neighborhood Search optimizer.

This module implements the Variable Neighborhood Search (VNS) optimizer. VNS is a
metaheuristic optimization algorithm that explores different neighborhoods of a
solution to find the optimal solution for a given objective function within a specified
search space.

The `VariableNeighborhoodSearch` class is the main class that implements the VNS
algorithm. It takes an objective function, lower and upper bounds of the search space,
dimensionality of the search space, and other optional parameters to control the
optimization process.

Example:
    ```python
    optimizer = VariableNeighborhoodSearch(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=1000,
        neighborhood_size=0.1,  # This is the size of the neighborhood for the shaking phase
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
    ```

Attributes:
    neighborhood_size (int): The size of the neighborhood for the shaking operation.
    population (np.ndarray): The population of individuals.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class VariableNeighborhoodSearch(AbstractOptimizer):
    """Implementation of the Variable Neighborhood Search optimizer.

    This optimizer uses the Variable Neighborhood Search (VNS) algorithm to find the optimal solution
    for a given objective function within a specified search space.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        neighborhood_size (int, optional): The size of the neighborhood. Defaults to 10.
        seed (Optional[int], optional): The seed value for random number generation. Defaults to None.


    Example:
        >>> from opt.metaheuristic.variable_neighbourhood_search import VariableNeighborhoodSearch
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = VariableNeighborhoodSearch(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = VariableNeighborhoodSearch(
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
        population_size: int = 100,
        max_iter: int = 1000,
        neighborhood_size: int = 10,
        seed: int | None = None,
    ) -> None:
        """Initialize the Variable Neighborhood Search optimizer."""
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

    def shaking(self, x: np.ndarray) -> np.ndarray:
        """Performs shaking operation on an individual by adding random noise to its coordinates.

        Args:
            x (np.ndarray): The individual to be shaken.

        Returns:
            np.ndarray: The shaken individual.

        """
        return x + np.random.default_rng(self.seed).uniform(
            -self.neighborhood_size, self.neighborhood_size, size=x.shape
        )

    def search(self) -> tuple[np.ndarray, float]:
        """Executes the Variable Neighborhood Search algorithm.

        This method performs the Variable Neighborhood Search algorithm to find the
        best individual within the search space that minimizes the objective function.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best individual found and
                its corresponding fitness value.
        """
        self.initialize_population()
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                x = self.population[i]
                x = self.shaking(x)
                x = np.clip(x, self.lower_bound, self.upper_bound)
                if self.func(x) < self.func(self.population[i]):
                    self.population[i] = x
        best_index = np.argmin(
            [self.func(individual) for individual in self.population]
        )
        return self.population[best_index], self.func(self.population[best_index])


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(VariableNeighborhoodSearch)
