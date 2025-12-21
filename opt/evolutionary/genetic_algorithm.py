"""Genetic Algorithm Optimizer.

This module implements a genetic algorithm (GA) optimizer. Genetic algorithms are a
part of evolutionary computing, which is a rapidly growing area of artificial
intelligence.

The GA optimizer starts with a population of candidate solutions to an optimization
problem and evolves this population by iteratively applying a set of genetic operators.

Key components of the GA optimizer include:
- Initialization: The population is initialized with a set of random solutions.
- Selection: Solutions are selected to reproduce based on their fitness. The better the
    solutions, the more chances they have to reproduce.
- Crossover (or recombination): Pairs of solutions are selected for reproduction to
    create one or more offspring, in which each offspring consists of a mix of the
    parents' traits.
- Mutation: After crossover, the offspring are mutated with a small probability.
    Mutation introduces small changes in the solutions, providing genetic diversity.
- Replacement: The population is updated to include the new, fitter solutions.

The GA optimizer is suitable for solving both constrained and unconstrained optimization
problems. It's particularly useful for problems where the search space is large and
complex, and where traditional optimization methods may not be applicable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class GeneticAlgorithm(AbstractOptimizer):
    """Genetic Algorithm optimizer.

    This optimizer uses a genetic algorithm to search for the optimal solution
    to a given optimization problem.

    Parameters:
    - func (Callable[[ndarray], float]): The objective function to be minimized.
    - lower_bound (float): The lower bound of the search space.
    - upper_bound (float): The upper bound of the search space.
    - dim (int): The dimensionality of the search space.
    - population_size (int, optional): The size of the population. Default is 100.
    - max_iter (int, optional): The maximum number of iterations. Default is 1000.
    - tournament_size (int, optional): The size of the tournament for selection. Default is 3.
    - crossover_rate (float, optional): The crossover rate. Default is 0.8.
    - seed (Optional[int], optional): The seed for the random number generator. Default is None.
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 150,
        max_iter: int = 1000,
        tournament_size: int = 3,
        crossover_rate: float = 0.7,
        seed: int | None = None,
    ) -> None:
        """Initialize the GeneticAlgorithm class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate

    def _initialize(self) -> np.ndarray:
        """Initializes the population with random values within the specified bounds.

        Returns:
            np.ndarray: The initialized population.
        """
        return np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Performs crossover between two parents to produce a child.

        Args:
            parent1 (np.ndarray): The first parent.
            parent2 (np.ndarray): The second parent.
            rng (np.random.Generator): Random number generator.

        Returns:
            np.ndarray: The child produced by crossover.
        """
        r = rng.random(self.dim)
        return np.where(r < self.crossover_rate, parent1, parent2)

    def _mutation(self, individual: np.ndarray, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
        """Mutates an individual with a certain probability.

        Args:
            individual (np.ndarray): The individual to be mutated.
            mutation_rate (float): The probability of mutation.
            rng (np.random.Generator): Random number generator.

        Returns:
            np.ndarray: The mutated individual.
        """
        r = rng.random(self.dim)
        mutation_strength = rng.uniform(0.8, 1.2, self.dim)  # More moderate mutation
        return np.where(
            r < mutation_rate,
            individual * mutation_strength,
            individual,
        )

    def _compute_mutation_rate(self, iteration: int) -> float:
        """Computes the mutation rate based on the current iteration.

        Args:
            iteration (int): The current iteration.

        Returns:
            float: The mutation rate.
        """
        return 0.5 * (1 + np.sin(iteration / self.max_iter * np.pi - np.pi / 2))

    def _selection(self, population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Selects an individual from the population based on fitness.

        The selection process is performed by converting the fitness values to probabilities,
        normalizing the probabilities, and then randomly choosing an individual based on the
        probabilities.

        Args:
            population (np.ndarray): The population.
            fitness (np.ndarray): The fitness values of the population.
            rng (np.random.Generator): Random number generator.

        Returns:
            np.ndarray: The selected individual.
        """
        # Shift fitness to ensure all values are positive, then invert for minimization
        shifted_fitness = fitness - np.min(fitness) + 1e-10
        selection_probs = 1 / shifted_fitness
        selection_probs /= np.sum(selection_probs)  # Normalize probabilities
        idx = rng.choice(np.arange(self.population_size), p=selection_probs)
        return population[idx].copy()

    def search(self) -> tuple[np.ndarray, float]:
        """Run the genetic algorithm search.

        Returns:
        - Tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        rng = np.random.default_rng(self.seed)
        population = self._initialize()
        best_solution: np.ndarray = np.zeros(self.dim)
        best_fitness = np.inf

        for i in range(self.max_iter):
            fitness = np.apply_along_axis(self.func, 1, population)

            # Track best solution (elitism)
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_fitness:
                best_fitness = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx].copy()

            new_population = []

            # Elitism: keep the best solution
            new_population.append(best_solution.copy())

            for _ in range(self.population_size - 1):
                parent1 = self._selection(population, fitness, rng)
                parent2 = self._selection(population, fitness, rng)
                child = self._crossover(parent1, parent2, rng)
                mutation_rate = self._compute_mutation_rate(i)
                child = self._mutation(child, mutation_rate, rng)
                # Clip to bounds
                child = np.clip(child, self.lower_bound, self.upper_bound)
                new_population.append(child)

            population = np.array(new_population)

        return best_solution, best_fitness


if __name__ == "__main__":
    optimizer = GeneticAlgorithm(
        func=shifted_ackley, lower_bound=-2.768, upper_bound=+2.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness: {best_fitness}")
