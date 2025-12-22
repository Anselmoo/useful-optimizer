"""Squirrel Search Algorithm.

!!! warning

    This module is still under development and is not yet ready for use.
"""

from __future__ import annotations

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


class SquirrelSearchAlgorithm(AbstractOptimizer):
    """Implementation of the Squirrel Search Algorithm.

    The Squirrel Search Algorithm is a population-based optimization algorithm
    inspired by the behavior of squirrels. It aims to find the optimal solution
    to a given optimization problem.

    Attributes:
        gliding_constant (float): The gliding constant used in the algorithm.
        max_gliding_distance (float): The maximum gliding distance allowed.
        min_gliding_distance (float): The minimum gliding distance allowed.
        seed (int): The seed value used for random number generation.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        population_size (int): The size of the population.
        dim (int): The dimensionality of the problem.
        max_iter (int): The maximum number of iterations.



    Example:
        >>> from opt.swarm_intelligence.squirrel_search import SquirrelSearchAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SquirrelSearchAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SquirrelSearchAlgorithm(
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

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Squirrel Search Algorithm and return the best solution.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found
            and its corresponding fitness value.

        """
        self.gliding_constant = 1.9
        self.max_gliding_distance = 1.11
        self.min_gliding_distance = 0.5
        # Initialize population
        squirrels = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Calculate fitness
        fitness = np.apply_along_axis(self.func, 1, squirrels)

        # Initialize the best solution
        best_fitness = np.min(fitness)
        best_squirrel = squirrels[np.argmin(fitness)]

        # Initialize velocity
        velocity = np.zeros((self.population_size, self.dim))
        random_flying_probability = 0.5

        # Main loop
        for _ in range(self.max_iter):
            self.seed += 1
            for i in range(self.population_size):
                self.seed += 1
                # Update gliding distance
                gliding_distance = (
                    self.min_gliding_distance
                    + (self.max_gliding_distance - self.min_gliding_distance)
                    * np.random.default_rng(self.seed).random()
                )

                # Update velocity and position
                velocity[i] = velocity[i] + gliding_distance * self.gliding_constant * (
                    squirrels[i] - best_squirrel
                )
                squirrels[i] = squirrels[i] + velocity[i]

                # Boundary check
                squirrels[i] = np.clip(squirrels[i], self.lower_bound, self.upper_bound)

                # Calculate fitness
                new_fitness = self.func(squirrels[i])

                # Update if new position is better
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_squirrel = squirrels[i]

                # Random flying
                if (
                    np.random.default_rng(self.seed).random()
                    > random_flying_probability
                ):
                    self.seed += 1
                    squirrels[i] = best_squirrel + 0.001 * np.mean(
                        np.random.default_rng(self.seed).random(self.dim)
                    )

        return best_squirrel, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(SquirrelSearchAlgorithm)
