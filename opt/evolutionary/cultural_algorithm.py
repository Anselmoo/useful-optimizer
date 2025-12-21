"""Cultural Algorithm implementation.

This module provides an implementation of the Cultural Algorithm optimizer. The
Cultural Algorithm is a population-based optimization algorithm that combines
individual learning (exploitation) with social learning (exploration) to search
for the best solution to a given optimization problem.

The CulturalAlgorithm class is the main class of this module. It inherits from the
AbstractOptimizer class and implements the search method to perform the Cultural
Algorithm search.

Example usage:
    optimizer = CulturalAlgorithm(
        func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    func (Callable[[ndarray], float]): The objective function to be minimized.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimensionality of the search space.
    population_size (int, optional): The size of the population. Defaults to 100.
    max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
    belief_space_size (int, optional): The size of the belief space. Defaults to 20.
    scaling_factor (float, optional): The scaling factor used in mutation. Defaults to 0.5.
    mutation_probability (float, optional): The probability of mutation. Defaults to 0.5.
    elitism (float, optional): The elitism factor. Defaults to 0.1.
    seed (int | None, optional): The random seed. Defaults to None.

Returns:
    tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class CulturalAlgorithm(AbstractOptimizer):
    """Cultural Algorithm optimizer.

    This optimizer uses a Cultural Algorithm to search for the best solution
    to a given optimization problem.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        belief_space_size (int, optional): The size of the belief space. Defaults to 20.
        scaling_factor (float, optional): The scaling factor used in mutation. Defaults to 0.5.
        mutation_probability (float, optional): The probability of mutation. Defaults to 0.5.
        elitism (float, optional): The elitism factor. Defaults to 0.1.

    Returns:
        tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 100,
        max_iter: int = 1000,
        belief_space_size: int = 20,
        scaling_factor: float = 0.5,
        mutation_probability: float = 0.5,
        elitism: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize the CulturalAlgorithm class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.belief_space_size = belief_space_size
        self.scaling_factor = scaling_factor
        self.mutation_probability = mutation_probability
        self.elitism = elitism

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the Cultural Algorithm search.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        # Initialize population and belief space
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        belief_space = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.belief_space_size, self.dim)
        )

        for _ in range(self.max_iter):
            self.seed += 1
            # Evaluate fitness of population
            fitness = np.apply_along_axis(self.func, 1, population)

            # Update belief space based on best individuals
            best_indices = fitness.argsort()[: self.belief_space_size]
            belief_space = population[best_indices]

            # Generate new population based on belief space
            new_population = population.copy()
            for i in range(self.population_size):
                self.seed += 1
                if (
                    np.random.default_rng(self.seed + 1).random() < self.elitism
                ):  # elitism: keep 10% of best individuals
                    continue
                parent = belief_space[
                    np.random.default_rng(self.seed + 2).choice(self.belief_space_size)
                ]
                if (
                    np.random.default_rng(self.seed + 3).random()
                    < self.mutation_probability
                ):  # differential evolution with 50% probability
                    a, b, c = population[
                        np.random.default_rng(self.seed + 4).choice(
                            self.population_size, 3, replace=False
                        )
                    ]
                    child = a + self.scaling_factor * (b - c)
                else:  # normal mutation
                    child = parent + np.random.default_rng(self.seed + 5).uniform(
                        -1, 1, self.dim
                    )
                child = np.clip(child, self.lower_bound, self.upper_bound)
                new_population[i] = child
            population = new_population

        best_index = fitness.argmin()
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(CulturalAlgorithm)
