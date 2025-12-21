"""Particle Swarm Optimization (PSO) algorithm implementation.

This module provides an implementation of the Particle Swarm Optimization (PSO) algorithm for solving optimization problems.
PSO is a population-based stochastic optimization algorithm inspired by the social behavior of bird flocking or fish schooling.

The main class in this module is `ParticleSwarm`, which represents the PSO algorithm. It takes an objective function, lower and upper bounds of the search space, dimensionality of the search space, and other optional parameters as input. The `search` method performs the PSO optimization and returns the best solution found.

Example usage:
    optimizer = ParticleSwarm(
        func=shifted_ackley,
        lower_bound=-32.768,
        upper_bound=+32.768,
        dim=2,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

Classes:
    - ParticleSwarm: Particle Swarm Optimization (PSO) algorithm for optimization problems.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley
from opt.constants import DEFAULT_MAX_ITERATIONS
from opt.constants import DEFAULT_POPULATION_SIZE
from opt.constants import PSO_COGNITIVE_COEFFICIENT
from opt.constants import PSO_INERTIA_WEIGHT
from opt.constants import PSO_SOCIAL_COEFFICIENT


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ParticleSwarm(AbstractOptimizer):
    """Particle Swarm Optimization (PSO) algorithm for optimization problems.

    This class implements the Particle Swarm Optimization algorithm, which is a
    population-based stochastic optimization algorithm inspired by the social behavior
    of bird flocking or fish schooling.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The number of particles in the swarm (default: 100).
        max_iter (int, optional): The maximum number of iterations (default: 1000).
        c1 (float, optional): The cognitive parameter (default: 1.5).
        c2 (float, optional): The social parameter (default: 1.5).
        w (float, optional): The inertia weight (default: 0.5).
        seed (int | None, optional): The seed for the random number generator (default: None).

    Attributes:
        c1 (float): The cognitive parameter.
        c2 (float): The social parameter.
        w (float): The inertia weight.

    Methods:
        search(): Perform the particle swarm optimization.

    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = DEFAULT_POPULATION_SIZE,
        max_iter: int = DEFAULT_MAX_ITERATIONS,
        c1: float = PSO_COGNITIVE_COEFFICIENT,
        c2: float = PSO_SOCIAL_COEFFICIENT,
        w: float = PSO_INERTIA_WEIGHT,
        seed: int | None = None,
        track_history: bool = False,
    ) -> None:
        """Initialize the ParticleSwarm class.

        Args:
            func (Callable[[ndarray], float]): The objective function to be minimized.
            lower_bound (float): The lower bound of the search space.
            upper_bound (float): The upper bound of the search space.
            dim (int): The dimensionality of the search space.
            population_size (int, optional): The number of particles in the swarm (default: 100).
            max_iter (int, optional): The maximum number of iterations (default: 1000).
            c1 (float, optional): The cognitive parameter (default: 1.5).
            c2 (float, optional): The social parameter (default: 1.5).
            w (float, optional): The inertia weight (default: 0.5).
            seed (int | None, optional): The seed for the random number generator (default: None).
            track_history (bool, optional): Whether to track optimization history for visualization (default: False).
        """
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
            track_history=track_history,
        )
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the particle swarm optimization.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best position found and its corresponding fitness value.
        """
        # Initialize population and fitness
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(self.func, 1, population)

        # Initialize velocity
        velocity = np.zeros((self.population_size, self.dim))

        # Initialize best position and fitness
        best_position = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        # Main loop
        for _ in range(self.max_iter):
            # Track history if enabled
            if self.track_history:
                self.history["best_fitness"].append(float(best_fitness))
                self.history["best_solution"].append(best_position.copy())
                self.history["population_fitness"].append(fitness.copy())
                self.history["population"].append(population.copy())

            self.seed += 1
            # Update velocity
            r1 = np.random.default_rng(self.seed + 1).random(
                (self.population_size, self.dim)
            )
            r2 = np.random.default_rng(self.seed + 2).random(
                (self.population_size, self.dim)
            )
            velocity = (
                self.w * velocity
                + self.c1 * r1 * (best_position - population)
                + self.c2 * r2 * (population[np.argmin(fitness)] - population)
            )

            # Update position
            population += velocity

            # Ensure the position stays within the bounds
            population = np.clip(population, self.lower_bound, self.upper_bound)

            # Update fitness
            fitness = np.apply_along_axis(self.func, 1, population)

            # Update best position and fitness
            best_index = np.argmin(fitness)
            if fitness[best_index] < best_fitness:
                best_position = population[best_index]
                best_fitness = fitness[best_index]

        # Track final state
        if self.track_history:
            self.history["best_fitness"].append(float(best_fitness))
            self.history["best_solution"].append(best_position.copy())
            self.history["population_fitness"].append(fitness.copy())
            self.history["population"].append(population.copy())

        return best_position, best_fitness


if __name__ == "__main__":
    optimizer = ParticleSwarm(
        func=shifted_ackley, lower_bound=-32.768, upper_bound=+32.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
