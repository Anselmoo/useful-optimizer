"""Memory-efficient history tracking for optimization algorithms.

This module provides classes for tracking optimization history with
pre-allocated NumPy arrays for better memory efficiency and cache locality
compared to Python lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from numpy import ndarray


@dataclass
class HistoryConfig:
    r"""Configuration for optimizer history tracking.

    Attributes:
        track_best_fitness (bool): Track best fitness value per iteration.
            Default: True.
        track_best_solution (bool): Track best solution vector per iteration.
            Default: True.
        track_population_fitness (bool): Track all population fitness values.
            Memory intensive: O(iterations $\times$ population_size).
            Default: False.
        track_population (bool): Track all population positions.
            Very memory intensive: O(iterations $\times$ population_size $\times$ dim).
            Default: False.
        max_history_size (int | None): Maximum number of iterations to track.
            If None, uses max_iter from optimizer.
            Default: None.
    """

    track_best_fitness: bool = True
    track_best_solution: bool = True
    track_population_fitness: bool = False
    track_population: bool = False
    max_history_size: int | None = None


class OptimizationHistory:
    """Memory-efficient history tracking with pre-allocated NumPy arrays.

    This class provides O(1) recording operations by pre-allocating arrays
    based on known iteration count and problem dimensions.

    Args:
        max_iter (int): Maximum number of iterations (for array allocation).
        dim (int): Dimensionality of the search space.
        population_size (int): Number of individuals in population (if applicable).
        config (HistoryConfig): Configuration for what to track.

    Attributes:
        best_fitness (ndarray | None): Array of best fitness values per iteration.
            Shape: (max_iter,).
        best_solution (ndarray | None): Array of best solution vectors per iteration.
            Shape: (max_iter, dim).
        population_fitness (ndarray | None): Array of all population fitness values.
            Shape: (max_iter, population_size).
        population (ndarray | None): Array of all population positions.
            Shape: (max_iter, population_size, dim).

    Notes:
        **Memory Efficiency**:
            For 10,000 iterations, dim=30, population=100:
            - best_fitness: 80 KB
            - best_solution: 2.4 MB
            - population: 240 MB (only if tracked)

        **Cache Efficiency**:
            Pre-allocated arrays provide contiguous memory layout
            for better CPU cache performance compared to Python lists.

    Example:
        >>> config = HistoryConfig(track_population=False)
        >>> history = OptimizationHistory(max_iter=100, dim=10, population_size=30, config=config)
        >>> # Record iteration 0
        >>> history.record(best_fitness=15.5, best_solution=np.random.rand(10))
        >>> # Export to dict
        >>> data = history.to_dict()
        >>> len(data["best_fitness"])
        1
    """

    def __init__(
        self,
        max_iter: int,
        dim: int,
        population_size: int,
        config: HistoryConfig | None = None,
    ) -> None:
        """Initialize history tracking with pre-allocated arrays."""
        self._iteration = 0
        self._max_iter = max_iter
        self._config = config or HistoryConfig()

        # Use max_history_size if provided, otherwise use max_iter
        history_size = self._config.max_history_size or max_iter

        # Pre-allocate arrays based on config
        self.best_fitness: ndarray | None = None
        self.best_solution: ndarray | None = None
        self.population_fitness: ndarray | None = None
        self.population: ndarray | None = None

        if self._config.track_best_fitness:
            self.best_fitness = np.full(history_size, np.inf, dtype=np.float64)

        if self._config.track_best_solution:
            self.best_solution = np.zeros((history_size, dim), dtype=np.float64)

        if self._config.track_population_fitness:
            self.population_fitness = np.zeros(
                (history_size, population_size), dtype=np.float64
            )

        if self._config.track_population:
            self.population = np.zeros(
                (history_size, population_size, dim), dtype=np.float64
            )

    def record(
        self,
        best_fitness: float,
        best_solution: ndarray,
        population_fitness: ndarray | None = None,
        population: ndarray | None = None,
    ) -> None:
        """Record one iteration's data. O(1) operation.

        Args:
            best_fitness (float): Best fitness value for this iteration.
            best_solution (ndarray): Best solution vector for this iteration.
                Shape: (dim,).
            population_fitness (ndarray | None): All population fitness values.
                Shape: (population_size,). Only used if track_population_fitness=True.
            population (ndarray | None): All population positions.
                Shape: (population_size, dim). Only used if track_population=True.

        Notes:
            This method silently ignores recording beyond max_iter to prevent
            index errors. The iteration counter will not increment.
        """
        if (
            self._iteration >= len(self.best_fitness)
            if self.best_fitness is not None
            else self._max_iter
        ):
            # Silently ignore recording beyond max_iter
            return

        i = self._iteration

        if self._config.track_best_fitness and self.best_fitness is not None:
            self.best_fitness[i] = best_fitness

        if self._config.track_best_solution and self.best_solution is not None:
            self.best_solution[i] = best_solution

        if (
            self._config.track_population_fitness
            and population_fitness is not None
            and self.population_fitness is not None
        ):
            self.population_fitness[i] = population_fitness

        if (
            self._config.track_population
            and population is not None
            and self.population is not None
        ):
            self.population[i] = population

        self._iteration += 1

    def to_dict(self) -> dict[str, list]:
        """Export to IOHprofiler-compatible format.

        Returns:
            dict: Dictionary with keys matching original history format:
                - "best_fitness": List of best fitness values
                - "best_solution": List of best solution vectors
                - "population_fitness": List of population fitness arrays (if tracked)
                - "population": List of population arrays (if tracked)

        Notes:
            Arrays are truncated to actual number of iterations recorded
            and converted to Python lists for JSON compatibility.
        """
        result: dict[str, list] = {}

        if self.best_fitness is not None:
            result["best_fitness"] = self.best_fitness[: self._iteration].tolist()

        if self.best_solution is not None:
            result["best_solution"] = self.best_solution[: self._iteration].tolist()

        if self.population_fitness is not None:
            result["population_fitness"] = self.population_fitness[
                : self._iteration
            ].tolist()

        if self.population is not None:
            result["population"] = self.population[: self._iteration].tolist()

        return result

    @property
    def iteration_count(self) -> int:
        """Get the number of iterations recorded.

        Returns:
            int: Number of iterations recorded so far.
        """
        return self._iteration

    def reset(self) -> None:
        """Reset history tracking to iteration 0.

        This resets the iteration counter but does not clear the arrays.
        Subsequent records will overwrite previous data.
        """
        self._iteration = 0
