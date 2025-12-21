"""Social Group Optimization Algorithm.

This module implements the Social Group Optimization (SGO) algorithm,
a social-inspired metaheuristic based on human social behavior.

The algorithm simulates social interaction behaviors including improving,
acquiring knowledge from others, and self-introspection.

Reference:
    Satapathy, S. C., & Naik, A. (2016).
    Social group optimization (SGO): A new population evolutionary optimization
    technique.
    Complex & Intelligent Systems, 2(3), 173-203.
    DOI: 10.1007/s40747-016-0022-8

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = SocialGroupOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-2.768,
    ...     upper_bound=2.768,
    ...     dim=2,
    ...     population_size=30,
    ...     max_iter=100,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class SocialGroupOptimizer(AbstractOptimizer):
    """Social Group Optimization algorithm.

    This algorithm simulates human social behaviors:
    1. Improving phase - learning from the best person
    2. Acquiring phase - learning from other group members
    3. Self-introspection - exploring individually

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of individuals in the social group.
        max_iter: Maximum number of iterations.
        c: Self-introspection coefficient.
        track_convergence: Whether to track fitness history.
        convergence_history: List of best fitness values per iteration (if tracking).
        early_stopping: Whether to enable early stopping.
        tolerance: Minimum improvement threshold for early stopping.
        patience: Number of iterations without improvement before stopping.
        verbose: Whether to print progress during optimization.


    Example:
        >>> from opt.social_inspired.social_group_optimizer import SocialGroupOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SocialGroupOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SocialGroupOptimizer(
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

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 30,
        max_iter: int = 100,
        c: float = 0.2,
        track_convergence: bool = False,
        early_stopping: bool = False,
        tolerance: float = 1e-6,
        patience: int = 10,
        verbose: bool = False,
    ) -> None:
        """Initialize Social Group Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of individuals. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
            c: Self-introspection coefficient. Defaults to 0.2.
            track_convergence: Enable convergence history tracking. Defaults to False.
            early_stopping: Enable early stopping. Defaults to False.
            tolerance: Minimum improvement threshold. Defaults to 1e-6.
            patience: Iterations without improvement before stopping. Defaults to 10.
            verbose: Print progress during optimization. Defaults to False.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.c = c
        self.track_convergence = track_convergence
        self.convergence_history: list[float] = []
        self.early_stopping = early_stopping
        self.tolerance = tolerance
        self.patience = patience
        self.verbose = verbose

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Social Group Optimization algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population (social group)
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Track convergence history if enabled
        if self.track_convergence:
            self.convergence_history.append(best_fitness)

        # Early stopping variables
        no_improvement_count = 0
        previous_best_fitness = best_fitness

        if self.verbose:
            print(f"Initial best fitness: {best_fitness:.6f}")

        for iteration in range(self.max_iter):
            # Update self-introspection coefficient
            c_current = self.c * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                new_position = population[i].copy()

                # Phase 1: Improving phase (learn from best)
                r1 = np.random.random(self.dim)
                improving_component = r1 * (best_solution - population[i])

                # Phase 2: Acquiring phase (learn from random member)
                j = np.random.randint(self.population_size)
                while j == i:
                    j = np.random.randint(self.population_size)

                r2 = np.random.random(self.dim)
                if fitness[j] < fitness[i]:
                    acquiring_component = r2 * (population[j] - population[i])
                else:
                    acquiring_component = r2 * (population[i] - population[j])

                # Phase 3: Self-introspection (individual exploration)
                r3 = np.random.uniform(-1, 1, self.dim)
                introspection_component = (
                    c_current * r3 * (self.upper_bound - self.lower_bound)
                )

                # Combine all phases
                new_position = (
                    population[i]
                    + improving_component
                    + acquiring_component
                    + introspection_component
                )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                new_fitness = self.func(new_position)

                # Greedy selection
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

            # Track convergence history if enabled
            if self.track_convergence:
                self.convergence_history.append(best_fitness)

            # Verbose progress reporting
            if self.verbose and (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}/{self.max_iter}: "
                    f"Best fitness = {best_fitness:.6f}"
                )

            # Early stopping check
            if self.early_stopping:
                improvement = previous_best_fitness - best_fitness
                # Only count iterations with minimal or no improvement
                if improvement >= 0 and improvement < self.tolerance:
                    no_improvement_count += 1
                    if no_improvement_count >= self.patience:
                        if self.verbose:
                            print(
                                f"Early stopping at iteration {iteration + 1}: "
                                f"No improvement for {self.patience} iterations"
                            )
                        break
                elif improvement >= self.tolerance:
                    # Significant improvement detected, reset counter
                    no_improvement_count = 0
                previous_best_fitness = best_fitness

        if self.verbose:
            print(f"Final best fitness: {best_fitness:.6f}")

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(SocialGroupOptimizer)
