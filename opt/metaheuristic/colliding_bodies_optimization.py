"""This module contains the implementation of the Colliding Bodies Optimization algorithm.

The Colliding Bodies Optimization algorithm is inspired by the behavior of colliding
bodies in physics. It aims to find the global minimum of a given objective function.

Example usage:
    optimizer = CollidingBodiesOptimization(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

"""

from __future__ import annotations

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


class CollidingBodiesOptimization(AbstractOptimizer):
    """Implementation of the Colliding Bodies Optimization algorithm.

    This optimizer is inspired by the behavior of colliding bodies in physics.
    It aims to find the global minimum of a given objective function.

    Attributes:
        step_size (float): The step size used for updating the positions of the agents.
        collision_radius (float): The collision radius used for calculating new velocities and positions.
        best_fitness (float): The best fitness value found during the optimization process.
        best_solution (np.ndarray): The best solution found during the optimization process.

    """

    def initialize_parameters(self) -> None:
        """Initialize the parameters of the optimizer."""
        self.step_size = 0.1 * (self.upper_bound - self.lower_bound)
        self.collision_radius = 0.1 * (self.upper_bound - self.lower_bound)
        self.best_fitness = np.inf
        self.best_solution = None

    def initialize_population(self) -> None:
        """Initialize the population of agents."""
        self.population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.population)

    def update_population(self) -> None:
        """Update the population of agents."""
        for _ in range(self.max_iter):
            self.seed += 1
            # Select two random agents for each agent in the population
            indices = np.random.default_rng(self.seed).choice(
                self.population_size, (self.population_size, 2), replace=True
            )
            a1, a2 = self.population[indices[:, 0]], self.population[indices[:, 1]]

            # Calculate new velocities and positions
            v1 = (a1 - a2) / (np.linalg.norm(a1 - a2, axis=1, keepdims=True) + 1e-7)
            v2 = (a2 - a1) / (np.linalg.norm(a2 - a1, axis=1, keepdims=True) + 1e-7)
            a1 += v1
            a2 += v2

            # Update population if new positions are better
            fitness_new = np.apply_along_axis(self.func, 1, a1)
            improved = fitness_new < self.fitness[indices[:, 0]]
            self.population[indices[improved, 0]] = a1[improved]
            self.fitness[indices[improved, 0]] = fitness_new[improved]

            fitness_new = np.apply_along_axis(self.func, 1, a2)
            improved = fitness_new < self.fitness[indices[:, 1]]
            self.population[indices[improved, 1]] = a2[improved]
            self.fitness[indices[improved, 1]] = fitness_new[improved]

    def search(self) -> tuple[np.ndarray, float]:
        """Run the optimization process and return the best solution found.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.

        """
        self.initialize_parameters()
        self.initialize_population()
        self.update_population()
        # Return the best solution
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]


if __name__ == "__main__":
    optimizer = CollidingBodiesOptimization(
        func=shifted_ackley, dim=2, lower_bound=-32.768, upper_bound=+32.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
