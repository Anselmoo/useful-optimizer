"""Bee Algorithm optimizer implementation.

This module provides an implementation of the Bee Algorithm optimizer.
The Bee Algorithm is a population-based optimization algorithm inspired
by the foraging behavior of honey bees. It is commonly used for solving
optimization problems.

The BeeAlgorithm class is the main class that implements the Bee Algorithm optimizer.
It takes an objective function, the dimensionality of the problem, and other optional
parameters as input. The search method runs the optimization process and returns the
best solution found and its corresponding fitness value.

Example usage:
    optimizer = BeeAlgorithm(
        func=shifted_ackley,
        dim=2,
        lower_bound=-2.768,
        upper_bound=+2.768,
        max_iter=4000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    population (np.ndarray): The current population of bees.
    fitness (np.ndarray): The fitness values of the population.
    prob (np.ndarray): The probability values for the onlooker bee phase.
    scout_bee (float): The probability of a bee becoming a scout bee.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class BeeAlgorithm(AbstractOptimizer):
    """Implementation of the Bee Algorithm optimizer.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        dim (int): The dimensionality of the problem.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        n_bees (int, optional): The number of bees in the population. Defaults to 50.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        scout_bee (float, optional): The probability of a bee becoming a scout bee. Defaults to 0.01.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Attributes:
        population (np.ndarray): The current population of bees.
        fitness (np.ndarray): The fitness values of the population.
        prob (np.ndarray): The probability values for the onlooker bee phase.
        scout_bee (float): The probability of a bee becoming a scout bee.

    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        dim: int,
        lower_bound: float,
        upper_bound: float,
        n_bees: int = 50,
        max_iter: int = 1000,
        scout_bee: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Initialize the BeeAlgorithm class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=n_bees,
        )

        self.population = np.random.default_rng(self.seed).uniform(
            lower_bound, upper_bound, (self.population_size, dim)
        )
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.prob = np.zeros(self.population_size)
        self.scout_bee = scout_bee

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Bee Algorithm optimization process.

        Returns:
            tuple[np.ndarray, float]: The best solution found and its corresponding fitness value.

        """
        for _ in range(self.max_iter):
            self.seed += 1
            # Employed Bee Phase
            for i in range(self.population_size):
                self.seed += 1
                candidate_solution = self.population[i] + np.random.default_rng(
                    self.seed
                ).uniform(-1, 1, self.dim)
                candidate_solution = np.clip(
                    candidate_solution, self.lower_bound, self.upper_bound
                )
                candidate_fitness = self.func(candidate_solution)
                if candidate_fitness < self.fitness[i]:
                    self.population[i] = candidate_solution
                    self.fitness[i] = candidate_fitness

            # Calculate probability values
            self.prob = (
                1.0
                - (self.fitness - np.min(self.fitness))
                / (np.max(self.fitness) - np.min(self.fitness))
            ) / self.population_size

            # Onlooker Bee Phase
            for i in range(self.population_size):
                self.seed += 1
                if np.random.default_rng(self.seed).random() < self.prob[i]:
                    candidate_solution = self.population[i] + np.random.default_rng(
                        self.seed + 1
                    ).uniform(-1, 1, self.dim)
                    candidate_solution = np.clip(
                        candidate_solution, self.lower_bound, self.upper_bound
                    )
                    candidate_fitness = self.func(candidate_solution)
                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate_solution
                        self.fitness[i] = candidate_fitness

            # Scout Bee Phase
            max_fitness_index = np.argmax(self.fitness)
            if (
                np.random.default_rng(self.seed).random() < self.scout_bee
            ):  # 1% chance to become scout bee
                self.population[max_fitness_index] = np.random.default_rng(
                    self.seed + 1
                ).uniform(self.lower_bound, self.upper_bound, self.dim)
                self.fitness[max_fitness_index] = self.func(
                    self.population[max_fitness_index]
                )

        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(BeeAlgorithm)
