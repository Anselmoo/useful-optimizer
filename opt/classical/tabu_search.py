"""Tabu Search.

This module implements the Tabu Search optimization algorithm.

The Tabu Search algorithm is a metaheuristic optimization algorithm that is used to
solve combinatorial optimization problems. It is inspired by the concept of memory in
human search behavior. The algorithm maintains a tabu list that keeps track of recently
visited solutions and prevents the search from revisiting them in the near future. This
helps the algorithm to explore different regions of the search space and avoid getting
stuck in local optima.

This module provides the `TabuSearch` class, which is an implementation of the
Tabu Search algorithm. It can be used to minimize a given objective function
over a continuous search space.

Example:
    ```python
    from opt.tabu_search import TabuSearch
    from opt.benchmark.functions import shifted_ackley


    # Define the objective function
    def objective_function(x):
        return shifted_ackley(x)


    # Create an instance of the TabuSearch optimizer
    optimizer = TabuSearch(
        func=objective_function,
        lower_bound=-32.768,
        upper_bound=32.768,
        dim=2,
        population_size=100,
        max_iter=1000,
        tabu_list_size=50,
        neighborhood_size=10,
        seed=None,
    )

    # Run the Tabu Search algorithm
    best_solution, best_fitness = optimizer.search()

    # Print the best solution and fitness value
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
    ```

Attributes:
    tabu_list_size (int): The size of the tabu list.
    neighborhood_size (int): The size of the neighborhood.
    population (ndarray | None): The current population.
    scores (ndarray | None): The scores of the current population.
    tabu_list (list): The tabu list.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class TabuSearch(AbstractOptimizer):
    """Tabu Search optimizer.

    This class implements the Tabu Search optimization algorithm.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        tabu_list_size (int, optional): The size of the tabu list. Defaults to 50.
        neighborhood_size (int, optional): The size of the neighborhood. Defaults to 10.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Attributes:
        tabu_list_size (int): The size of the tabu list.
        neighborhood_size (int): The size of the neighborhood.
        population (ndarray | None): The current population.
        scores (ndarray | None): The scores of the current population.
        tabu_list (list): The tabu list.



    Example:
        >>> from opt.classical.tabu_search import TabuSearch
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = TabuSearch(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = TabuSearch(
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
        tabu_list_size: int = 50,
        neighborhood_size: int = 10,
        seed: int | None = None,
    ) -> None:
        """Initialize the TabuSearch class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.tabu_list_size = tabu_list_size
        self.neighborhood_size = neighborhood_size
        self.population: ndarray = np.empty((self.population_size, self.dim))
        self.scores = np.empty(self.population_size)
        self.tabu_list: list = []

    def initialize_population(self) -> None:
        """Initialize the population.

        This method initializes the population by generating random individuals within the search space.
        """
        self.population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        self.scores = np.array(
            [self.func(individual) for individual in self.population]
        )

    def generate_neighborhood(self, solution: ndarray) -> ndarray:
        """Generate the neighborhood of a solution.

        This method generates a neighborhood of solutions by perturbing the given solution.

        Args:
            solution (ndarray): The solution to generate the neighborhood for.

        Returns:
            ndarray: The generated neighborhood.

        """
        neighborhood = [
            solution + np.random.default_rng(self.seed).uniform(-0.1, 0.1, self.dim)
            for _ in range(self.neighborhood_size)
        ]
        return np.clip(neighborhood, self.lower_bound, self.upper_bound)

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Tabu Search algorithm.

        This method performs the Tabu Search algorithm to find the best solution.

        Returns:
            tuple[np.ndarray, float]: The best solution found and its corresponding score.

        """
        self.initialize_population()
        for _ in range(self.max_iter):
            best_index = np.argmin(self.scores)
            if len(self.tabu_list) >= self.tabu_list_size:
                self.tabu_list.pop(0)
            self.tabu_list.append(self.population[best_index])
            neighborhood = self.generate_neighborhood(self.population[best_index])
            neighborhood_scores = np.array(
                [self.func(individual) for individual in neighborhood]
            )
            best_neighbor_index = np.argmin(neighborhood_scores)
            if not any(
                np.array_equal(x, neighborhood[best_neighbor_index])
                for x in self.tabu_list
            ):
                self.population[best_index] = neighborhood[best_neighbor_index]
                self.scores[best_index] = neighborhood_scores[best_neighbor_index]
        best_index = np.argmin(self.scores)
        return self.population[best_index], self.scores[best_index]


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(TabuSearch)
