"""RIME Optimization Algorithm.

This module implements the RIME optimization algorithm, a physics-based
metaheuristic inspired by the natural phenomenon of rime-ice formation.

Rime is a type of ice formed when supercooled water droplets freeze on
contact with a surface. The algorithm simulates this physical process
for optimization.

Reference:
    Su, H., Zhao, D., Heidari, A. A., Liu, L., Zhang, X., Mafarja, M., &
    Chen, H. (2023).
    RIME: A physics-based optimization.
    Neurocomputing, 532, 183-214.
    DOI: 10.1016/j.neucom.2023.02.010

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = RIMEOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-2.768,
    ...     upper_bound=2.768,
    ...     dim=2,
    ...     population_size=30,
    ...     max_iter=100
    ... )
    >>> best_solution, best_fitness = optimizer.search()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class RIMEOptimizer(AbstractOptimizer):
    """RIME Optimization Algorithm optimizer.

    This physics-based algorithm simulates rime-ice formation:
    1. Soft-rime search - exploration phase with large perturbations
    2. Hard-rime puncture - exploitation with focused search
    3. Positive greedy selection - accepting improvements

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of agents in the population.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.physics_inspired.rime_optimizer import RIMEOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = RIMEOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = RIMEOptimizer(
        ...     func=shifted_ackley,
        ...     dim=2,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     max_iter=10
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
    ) -> None:
        """Initialize RIME Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of agents. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the RIME Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Rime-ice factor decreases over iterations
            rime_factor = (1 - iteration / self.max_iter) ** 5

            for i in range(self.population_size):
                new_position = population[i].copy()

                # Soft-rime search strategy (exploration)
                for j in range(self.dim):
                    r1 = np.random.random()
                    if r1 < rime_factor:
                        # Soft-rime update based on best solution
                        h = 2 * rime_factor * np.random.random() - rime_factor
                        new_position[j] = best_solution[j] + h * (
                            best_solution[j]
                            - population[i][j]
                            * np.random.random()
                            * (self.upper_bound - self.lower_bound)
                            * 0.1
                        )

                # Hard-rime puncture strategy (exploitation)
                r2 = np.random.random()
                e = np.sqrt(iteration / self.max_iter)

                if r2 < e:
                    # Select random dimensions to update
                    num_dims = np.random.randint(1, self.dim + 1)
                    dims_to_update = np.random.choice(self.dim, num_dims, replace=False)

                    for j in dims_to_update:
                        # Puncture toward normalized best position
                        normalized_best = (best_solution[j] - self.lower_bound) / (
                            self.upper_bound - self.lower_bound
                        )
                        new_position[j] = best_solution[j] - normalized_best * (
                            self.upper_bound - self.lower_bound
                        ) * (2 * np.random.random() - 1) * (
                            1 - iteration / self.max_iter
                        )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                new_fitness = self.func(new_position)

                # Positive greedy selection
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(RIMEOptimizer)
