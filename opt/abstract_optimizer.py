"""Abstract base class for single-objective optimizers.

**COCO/BBOB Compliance Requirements:**
All concrete optimizer implementations inheriting from this class must provide:
- Algorithm metadata (name, version, authors, year, class)
- BBOB benchmark settings (search space, dimensions, runs, seeds)
- Hyperparameter documentation with BBOB-recommended values
- Reproducibility requirements (seed logging, parameter tracking)
- Performance characteristics on BBOB function classes
- Complexity analysis (time/space, function evaluations)

See `.github/prompts/optimizer-docs-template.prompt.md` for complete template.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from opt.abstract.history import HistoryConfig
from opt.abstract.history import OptimizationHistory
from opt.constants import DEFAULT_MAX_ITERATIONS
from opt.constants import DEFAULT_POPULATION_SIZE
from opt.constants import DEFAULT_SEED
from opt.constants import POWER_THIRTY_TWO


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class AbstractOptimizer(ABC):
    """An abstract base class for optimizers with COCO/BBOB compliance support.

    This base class provides the foundation for single-objective optimization algorithms
    with built-in support for COCO/BBOB benchmark requirements including reproducibility,
    history tracking, and standardized interfaces.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
            Must accept a NumPy array and return a scalar.
            BBOB functions available in `opt.benchmark.functions`.
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
            If None, generates a random seed. Defaults to None.
        population_size (int, optional): The number of individuals in the population.
            BBOB recommendation: 10*dim for population-based algorithms.
            Defaults to 100.
        track_history (bool, optional): Whether to track optimization history.
            When enabled, stores convergence data for visualization and COCO postprocessing.
            Defaults to False.

    Attributes:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int): The maximum number of iterations for the optimization process.
        seed (int): **REQUIRED for BBOB compliance.** The random seed.
            Used for all random operations to ensure reproducibility.
        population_size (int): The number of individuals in the population.
        track_history (bool): Whether to track optimization history.
        history (dict[str, list]): Dictionary containing optimization history if track_history is True.
            Contains keys: 'best_fitness', 'best_solution', 'population_fitness', 'population'.

    Methods:
        search() -> tuple[ndarray, float]: Perform the optimization search.

    Returns:
        tuple[ndarray, float]: Tuple containing the best solution found (shape: (dim,))
        and its corresponding fitness value (scalar).

    Notes:
        **BBOB Standard Settings:**
        - Search space bounds: Typically [-5, 5] for most functions
        - Evaluation budget: dim * 10000 function evaluations
        - Independent runs: 15 (using seeds 0-14)
        - Target precision: 1e-8

        **Reproducibility:**
            - Same seed must produce identical results across runs
            - All random operations use `np.random.default_rng(self.seed)`
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = DEFAULT_MAX_ITERATIONS,
        seed: int | None = None,
        population_size: int = DEFAULT_POPULATION_SIZE,
        track_history: bool = False,
    ) -> None:
        """Initialize the optimizer."""
        self.func = func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.max_iter = max_iter
        if seed is None:
            self.seed = np.random.default_rng(DEFAULT_SEED).integers(
                0, 2**POWER_THIRTY_TWO
            )
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
                "population_fitness": [],
                "population": [],
            }
            if track_history
            else {}
        )

    def _record_history(
        self,
        best_fitness: float,
        best_solution: ndarray,
        population_fitness: ndarray | None = None,
        population: ndarray | None = None,
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

    def _finalize_history(self) -> None:
        """Convert preallocated history to list-based format for consumers."""
        if not self.track_history or self._history_buffer is None:
            return
        self.history = self._history_buffer.to_dict()

    @abstractmethod
    def search(self) -> tuple[ndarray, float]:
        """Perform the optimization search.

        Returns:
        Tuple containing the best solution found and its corresponding fitness value.
        """
