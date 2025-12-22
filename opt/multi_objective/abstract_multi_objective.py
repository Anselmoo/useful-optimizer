"""Abstract base class for multi-objective optimizers.

This module defines the base class for multi-objective optimization algorithms
that return Pareto-optimal solution sets instead of a single optimal solution.

References:
    Deb, K. et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm:
    NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from numpy import ndarray


class AbstractMultiObjectiveOptimizer(ABC):
    """Abstract base class for multi-objective optimizers.

    Multi-objective optimizers find a set of Pareto-optimal solutions that
    represent trade-offs between multiple competing objectives.

    Args:
        objectives: List of objective functions to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to None.
        population_size: Number of individuals in the population. Defaults to 100.

    Attributes:
        objectives: List of objective functions to minimize.
        num_objectives: Number of objectives.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        seed: Random seed for reproducibility.
        population_size: Number of individuals in the population.

    Example:
        >>> from opt.multi_objective.nsga_ii import NSGAII
        >>> import numpy as np
        >>> def f1(x):
        ...     return sum(x**2)
        >>> def f2(x):
        ...     return sum((x - 2) ** 2)
        >>> optimizer = NSGAII(
        ...     objectives=[f1, f2], lower_bound=-5, upper_bound=5, dim=3, max_iter=10
        ... )
        >>> pareto_front, pareto_fitness = optimizer.search()
        >>> isinstance(pareto_front, np.ndarray)
        True
    """

    def __init__(
        self,
        objectives: Sequence[Callable[[ndarray], float]],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 100,
    ) -> None:
        """Initialize the multi-objective optimizer."""
        self.objectives = list(objectives)
        self.num_objectives = len(objectives)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.max_iter = max_iter
        if seed is None:
            self.seed = np.random.default_rng(42).integers(0, 2**32)
        else:
            self.seed = seed
        self.population_size = population_size

    def evaluate(self, solution: ndarray) -> ndarray:
        """Evaluate a solution on all objectives.

        Args:
            solution: A candidate solution vector.

        Returns:
            Array of objective values for the solution.
        """
        return np.array([obj(solution) for obj in self.objectives])

    def evaluate_population(self, population: ndarray) -> ndarray:
        """Evaluate all solutions in a population.

        Args:
            population: 2D array of shape (population_size, dim).

        Returns:
            2D array of shape (population_size, num_objectives).
        """
        return np.array([self.evaluate(ind) for ind in population])

    @staticmethod
    def dominates(fitness_a: ndarray, fitness_b: ndarray) -> bool:
        """Check if solution A dominates solution B (minimization).

        A dominates B if A is no worse in all objectives and strictly
        better in at least one objective.

        Args:
            fitness_a: Objective values for solution A.
            fitness_b: Objective values for solution B.

        Returns:
            True if A dominates B, False otherwise.
        """
        return bool(np.all(fitness_a <= fitness_b) and np.any(fitness_a < fitness_b))

    def fast_non_dominated_sort(self, fitness: ndarray) -> list[list[int]]:
        """Perform fast non-dominated sorting.

        Args:
            fitness: 2D array of shape (population_size, num_objectives).

        Returns:
            List of fronts, where each front is a list of solution indices.
        """
        n = len(fitness)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions: list[list[int]] = [[] for _ in range(n)]
        fronts: list[list[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self.dominates(fitness[i], fitness[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(fitness[j], fitness[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # First front: solutions not dominated by anyone
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front: list[int] = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    @staticmethod
    def crowding_distance(fitness: ndarray, front: list[int]) -> ndarray:
        """Calculate crowding distance for solutions in a front.

        Args:
            fitness: 2D array of all fitness values.
            front: List of indices for solutions in this front.

        Returns:
            Array of crowding distances for each solution in the front.
        """
        n = len(front)
        _min_front_size = 2  # Minimum size for meaningful crowding distance
        if n <= _min_front_size:
            return np.full(n, np.inf)

        distances = np.zeros(n)
        front_fitness = fitness[front]

        for m in range(fitness.shape[1]):
            sorted_indices = np.argsort(front_fitness[:, m])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            f_range = (
                front_fitness[sorted_indices[-1], m]
                - front_fitness[sorted_indices[0], m]
            )
            if f_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (
                        front_fitness[sorted_indices[i + 1], m]
                        - front_fitness[sorted_indices[i - 1], m]
                    ) / f_range

        return distances

    @abstractmethod
    def search(self) -> tuple[ndarray, ndarray]:
        """Perform the multi-objective optimization search.

        Returns:
            Tuple containing:
                - pareto_solutions: 2D array of Pareto-optimal solutions
                  with shape (num_pareto_solutions, dim).
                - pareto_fitness: 2D array of objective values for each
                  Pareto solution with shape (num_pareto_solutions, num_objectives).
        """
