"""Glowworm Swarm Optimization (GSO) algorithm.

This module implements the Glowworm Swarm Optimization (GSO) algorithm as an optimizer.
GSO is a population-based optimization algorithm inspired by the behavior of glowworms.
It is commonly used to solve optimization problems.

The GlowwormSwarmOptimization class provides an implementation of the GSO algorithm. It
takes an objective function, lower and upper bounds of the search space, dimensionality
of the search space, and other optional parameters as input. The algorithm searches for
the best solution within the given search space by iteratively updating the positions of
glowworms based on their luciferin levels and neighboring glowworms.

Usage:
    optimizer = GlowwormSwarmOptimization(
        func=shifted_ackley, dim=2, lower_bound=-32.768, upper_bound=+32.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    luciferin_decay (float): The decay rate of luciferin.
    randomness (float): The randomness factor for glowworm movement.
    step_size (float): The step size for glowworm movement.

Methods:
    _initialize(): Initialize the population of glowworms.
    _compute_fitness(population): Compute the fitness values for the glowworm population.
    _update_luciferin(population, fitness): Update the luciferin levels of the glowworms.
    _move_glowworms(population, luciferin): Move the glowworms based on their luciferin levels.
    search(): Run the glowworm swarm optimization algorithm and return the best solution and fitness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class GlowwormSwarmOptimization(AbstractOptimizer):
    """Glowworm Swarm Optimization (GSO) algorithm.

    Glowworm Swarm Optimization (GSO) is a population-based optimization algorithm inspired by the behavior of glowworms.
    This class implements the GSO algorithm as an optimizer.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The number of glowworms in the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        luciferin_decay (float, optional): The decay rate of luciferin. Defaults to 0.1.
        randomness (float, optional): The randomness factor for glowworm movement. Defaults to 0.5.
        step_size (float, optional): The step size for glowworm movement. Defaults to 0.01.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Attributes:
        luciferin_decay (float): The decay rate of luciferin.
        randomness (float): The randomness factor for glowworm movement.
        step_size (float): The step size for glowworm movement.

    Methods:
        _initialize(): Initialize the population of glowworms.
        _compute_fitness(population): Compute the fitness values for the glowworm population.
        _update_luciferin(population, fitness): Update the luciferin levels of the glowworms.
        _move_glowworms(population, luciferin): Move the glowworms based on their luciferin levels.
        search(): Run the glowworm swarm optimization algorithm and return the best solution and fitness.



    Example:
        >>> from opt.swarm_intelligence.glowworm_swarm_optimization import GlowwormSwarmOptimization
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = GlowwormSwarmOptimization(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = GlowwormSwarmOptimization(
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
        luciferin_decay: float = 0.1,
        randomness: float = 0.5,
        step_size: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Initialize the GlowwormSwarmOptimization class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.luciferin_decay = luciferin_decay
        self.randomness = randomness
        self.step_size = step_size

    def _initialize(self) -> ndarray:
        """Initializes the population of glowworms with random positions.

        Returns:
            ndarray: The initialized population of glowworms.
        """
        return np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

    def _compute_fitness(self, population: ndarray) -> ndarray:
        """Compute the fitness values for the given population.

        Parameters:
        population (ndarray): The population for which to compute the fitness values.

        Returns:
        ndarray: The computed fitness values for the population.
        """
        return np.apply_along_axis(self.func, 1, population)

    def _update_luciferin(self, population: ndarray, fitness: ndarray) -> ndarray:
        """Update the luciferin levels of the glowworms based on their fitness values.

        Args:
            population (ndarray): The population of glowworms.
            fitness (ndarray): The fitness values of the glowworms.

        Returns:
            ndarray: The updated luciferin levels of the glowworms.
        """
        return (1 - self.luciferin_decay) * fitness / np.linalg.norm(population, axis=1)

    def _move_glowworms(self, population: ndarray, luciferin: ndarray) -> ndarray:
        """Moves the glowworms in the population based on their luciferin levels and neighboring glowworms.

        Args:
            population (ndarray): The current population of glowworms.
            luciferin (ndarray): The luciferin levels of the glowworms.

        Returns:
            ndarray: The new population of glowworms after moving.

        """
        new_population = []
        for i in range(self.population_size):
            self.seed += 1
            glowworm = population[i]
            neighbors = population[luciferin > luciferin[i]]
            if len(neighbors) > 0:
                distances = np.linalg.norm(neighbors - glowworm, axis=1)
                probabilities = luciferin[luciferin > luciferin[i]] / distances
                probabilities /= np.sum(probabilities)
                selected_neighbor = neighbors[
                    np.random.default_rng(self.seed).choice(
                        len(neighbors), p=probabilities
                    )
                ]
                direction = selected_neighbor - glowworm
                random_vector = np.random.default_rng(self.seed).uniform(
                    -1, 1, self.dim
                )
                glowworm += self.step_size * (
                    direction + self.randomness * random_vector
                )
            new_population.append(glowworm)
        return np.array(new_population)

    def search(self) -> tuple[np.ndarray, float]:
        """Run the glowworm swarm optimization algorithm and return the best solution and fitness.

        Returns:
            tuple[np.ndarray, float]: The best solution found by the algorithm and its corresponding fitness value.

        """
        population = self._initialize()
        best_solution = None
        best_fitness = np.inf
        for _ in range(self.max_iter):
            fitness = self._compute_fitness(population)
            luciferin = self._update_luciferin(population, fitness)
            population = self._move_glowworms(population, luciferin)
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_fitness:
                best_fitness = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]
        return best_solution, best_fitness


if __name__ == "__main__":
    optimizer = GlowwormSwarmOptimization(
        func=shifted_ackley, dim=2, lower_bound=-32.768, upper_bound=+32.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
