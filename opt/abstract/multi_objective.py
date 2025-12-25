"""Abstract base class for multi-objective optimizers.

This module defines the base class for multi-objective optimization algorithms
that return Pareto-optimal solution sets instead of a single optimal solution.

**COCO/BBOB Multi-Objective Compliance Requirements:**
All concrete multi-objective optimizer implementations must provide:
- Algorithm metadata (name, version, authors, year, class)
- Multi-objective BBOB benchmark settings (search space, dimensions, runs, seeds)
- Hyperparameter documentation with BBOB-recommended values
- Pareto front reproducibility requirements (seed logging, deterministic sorting)
- Multi-objective performance indicators (Hypervolume, IGD, Spread, Epsilon)
- Complexity analysis (time/space, function evaluations)

See `.github/prompts/optimizer-docs-template.prompt.md` for complete template.

References:
    Deb, K. et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm:
    NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from opt.abstract.history import HistoryConfig
from opt.abstract.history import OptimizationHistory


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from numpy import ndarray


class AbstractMultiObjectiveOptimizer(ABC):
    """Abstract base class for multi-objective optimizers with COCO/BBOB compliance support.

    Multi-objective optimizers find a set of Pareto-optimal solutions that
    represent trade-offs between multiple competing objectives. This base class
    provides built-in support for COCO/BBOB multi-objective benchmark requirements.

    Args:
        objectives (Sequence[Callable[[ndarray], float]]): Objective functions to minimize.
            Each function must accept a NumPy array and return a scalar.
            BBOB multi-objective test suites available.
        lower_bound (float): The lower bound of the search space.
            BBOB typical: -5 (most functions), -100 (Rastrigin, Weierstrass).
        upper_bound (float): The upper bound of the search space.
            BBOB typical: 5 (most functions), 100 (Rastrigin, Weierstrass).
        dim (int): The dimensionality of the search space.
            BBOB standard dimensions: 2, 3, 5, 10, 20, 40.
        max_iter (int, optional): The maximum number of iterations.
            BBOB recommendation: 10000 for complete evaluation.
            Defaults to 1000.
        seed (int | None, optional): **REQUIRED for BBOB compliance.** Random seed.
            BBOB requires seeds 0-14 for 15 independent runs.
            Ensures deterministic Pareto front generation.
            If None, generates a random seed. Defaults to None.
        population_size (int, optional): The number of individuals in the population.
            BBOB recommendation: 10*dim for population-based algorithms.
            Defaults to 100.

    Attributes:
        objectives (list[Callable[[ndarray], float]]): Objective functions to minimize.
        num_objectives (int): The number of objectives.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int): The maximum number of iterations.
        seed (int): **REQUIRED for BBOB compliance.** Random seed for reproducibility.
            Used for all random operations to ensure deterministic Pareto fronts.
        population_size (int): The number of individuals in the population.
        track_history (bool): Whether to record optimization history for analysis.
        history (dict[str, list]): Optimization history if track_history is True.
            Contains keys:
            - 'best_fitness': Scalarized best fitness per iteration.
            - 'best_solution': Corresponding solution vector.
            - 'pareto_fitness': Objective values of the non-dominated set.
            - 'pareto_solutions': Non-dominated solutions.
            - 'population_fitness': Fitness of full population per iteration.
            - 'population': Population positions per iteration.

    Methods:
        search() -> tuple[ndarray, ndarray]: Perform the multi-objective optimization search.

    Returns:
        tuple[ndarray, ndarray]: Tuple containing the Pareto-optimal solutions and fitness values.
        - pareto_solutions: 2D array of Pareto-optimal solutions with shape
        (num_pareto_solutions, dim).
        - pareto_fitness: 2D array of objective values for each Pareto solution with shape
        (num_pareto_solutions, num_objectives).

    Notes:
        **BBOB Multi-Objective Standard Settings:**
        - Search space bounds: Typically [-5, 5] for most functions
        - Evaluation budget: dim * 10000 function evaluations
        - Independent runs: 15 (using seeds 0-14)
        - Performance indicators: Hypervolume, IGD, Spread, Epsilon

        **Pareto Front Reproducibility:**
            - Same seed must produce identical Pareto fronts across runs
            - All random operations use `np.random.default_rng(self.seed)`
            - Deterministic tie-breaking in non-dominated sorting
            - Consistent ordering of solutions in returned Pareto front

        **Multi-Objective Performance Metrics:**
            - **Hypervolume (HV)**: Volume of objective space dominated by Pareto front
            - **Inverted Generational Distance (IGD)**: Distance to reference Pareto front
            - **Spread**: Diversity measure of Pareto front distribution
            - **Epsilon Indicator**: Multiplicative convergence metric

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
        track_history: bool = False,  # noqa: FBT001, FBT002
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
        self.track_history = track_history
        self._history_buffer = (
            OptimizationHistory(
                max_iter=self.max_iter + 1,  # include final state
                dim=self.dim,
                population_size=self.population_size,
                config=HistoryConfig(
                    track_population=True,
                    track_population_fitness=True,
                    max_history_size=self.max_iter + 1,
                ),
            )
            if track_history
            else None
        )
        self.history: dict[str, list] = (
            {
                "best_fitness": [],
                "best_solution": [],
                "pareto_fitness": [],
                "pareto_solutions": [],
                "population_fitness": [],
                "population": [],
            }
            if track_history
            else {}
        )

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

    def _record_history(
        self,
        best_fitness: float,
        best_solution: ndarray,
        population_fitness: ndarray | None = None,
        population: ndarray | None = None,
        pareto_fitness: ndarray | None = None,
        pareto_solutions: ndarray | None = None,
    ) -> None:
        """Record iteration history using preallocated storage."""
        if not self.track_history or self._history_buffer is None:
            return

        self._history_buffer.record(
            best_fitness=best_fitness,
            best_solution=best_solution,
            population_fitness=population_fitness,
            population=population,
        )

        if pareto_fitness is not None and pareto_solutions is not None:
            self.history["pareto_fitness"].append(pareto_fitness.copy())
            self.history["pareto_solutions"].append(pareto_solutions.copy())

    def _finalize_history(self) -> None:
        """Convert preallocated history to list-based format for consumers."""
        if not self.track_history or self._history_buffer is None:
            return

        buffer_dict = self._history_buffer.to_dict()

        if "pareto_fitness" in self.history:
            buffer_dict["pareto_fitness"] = self.history["pareto_fitness"]
        if "pareto_solutions" in self.history:
            buffer_dict["pareto_solutions"] = self.history["pareto_solutions"]

        self.history = buffer_dict

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
