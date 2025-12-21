"""Teaching-Learning Based Optimization (TLBO).

This module implements Teaching-Learning Based Optimization,
a metaheuristic algorithm inspired by the teaching-learning
process in a classroom.

Reference:
    Rao, R. V., Savsani, V. J., & Vakharia, D. P. (2011).
    Teaching-learning-based optimization: A novel method for constrained
    mechanical design optimization problems.
    Computer-Aided Design, 43(3), 303-315.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for TLBO algorithm
_TEACHING_FACTOR_MIN = 1
_TEACHING_FACTOR_MAX = 2


class TeachingLearningOptimizer(AbstractOptimizer):
    """Teaching-Learning Based Optimization implementation.

    TLBO simulates the teaching-learning process with two phases:
    1. Teacher Phase: Students learn from the teacher (best solution)
    2. Learner Phase: Students learn from interaction with each other

    A key feature of TLBO is that it has no algorithm-specific parameters
    to tune, only population size and iterations.

    Attributes:
        func: The objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        population_size: Number of students (learners).
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.social_inspired.teaching_learning import TeachingLearningOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = TeachingLearningOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = TeachingLearningOptimizer(
        ...     func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=2.768, max_iter=10
        ... )
        >>> _, fitness = optimizer.search()
        >>> isinstance(float(fitness), float)
        True
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 50,
        max_iter: int = 500,
    ) -> None:
        """Initialize the TLBO optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound for all dimensions.
            upper_bound: Upper bound for all dimensions.
            dim: Number of dimensions.
            population_size: Number of learners.
            max_iter: Maximum iterations.
        """
        super().__init__(func, lower_bound, upper_bound, dim)
        self.population_size = population_size
        self.max_iter = max_iter

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the optimization algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population (students)
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Initialize best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for _ in range(self.max_iter):
            # Calculate mean of population
            mean_population = np.mean(population, axis=0)

            # Teacher is the best solution
            teacher = best_solution.copy()

            # Teaching factor (randomly 1 or 2)
            teaching_factor = np.random.randint(
                _TEACHING_FACTOR_MIN, _TEACHING_FACTOR_MAX + 1
            )

            # ===== Teacher Phase =====
            for i in range(self.population_size):
                # Difference mean
                diff_mean = np.random.rand(self.dim) * (
                    teacher - teaching_factor * mean_population
                )

                # New position after learning from teacher
                new_position = population[i] + diff_mean

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update if better (greedy selection)
                new_fitness = self.func(new_position)
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    # Update best if necessary
                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

            # ===== Learner Phase =====
            for i in range(self.population_size):
                # Randomly select another learner
                j = np.random.randint(self.population_size)
                while j == i:
                    j = np.random.randint(self.population_size)

                # Learn from the better learner
                if fitness[i] < fitness[j]:
                    # Current learner is better
                    new_position = population[i] + np.random.rand(self.dim) * (
                        population[i] - population[j]
                    )
                else:
                    # Other learner is better
                    new_position = population[i] + np.random.rand(self.dim) * (
                        population[j] - population[i]
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update if better (greedy selection)
                new_fitness = self.func(new_position)
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    # Update best if necessary
                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(TeachingLearningOptimizer)
