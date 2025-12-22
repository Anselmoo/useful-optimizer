"""Abstract base class for single-objective optimizers.

**COCO/BBOB Compliance Requirements:**
All concrete optimizer implementations inheriting from this class must provide:
- Algorithm metadata (name, version, authors, year, class)
- BBOB benchmark settings (search space, dimensions, runs, seeds)
- Hyperparameter documentation with BBOB-recommended values
- Reproducibility requirements (seed logging, parameter tracking)
- Performance characteristics on BBOB function classes
- Complexity analysis (time/space, function evaluations)

See `.github/prompts/optimizer-docs-template.md` for complete template.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

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
        func (Callable[[ndarray], float]):
            The objective function to be optimized. Must accept numpy array and return scalar.
            BBOB functions available in `opt.benchmark.functions`.
        lower_bound (float):
            The lower bound of the search space.
            BBOB typical: -5 (most functions), -100 (Rastrigin, Weierstrass).
        upper_bound (float):
            The upper bound of the search space.
            BBOB typical: 5 (most functions), 100 (Rastrigin, Weierstrass).
        dim (int):
            The dimensionality of the search space.
            BBOB standard dimensions: 2, 3, 5, 10, 20, 40.
        max_iter (int, optional):
            The maximum number of iterations for the optimization process.
            BBOB recommendation: 10000 for complete evaluation.
            Defaults to 1000.
        seed (int | None, optional):
            **REQUIRED for BBOB compliance.** Random seed for reproducibility.
            BBOB requires seeds 0-14 for 15 independent runs.
            If None, generates random seed. Defaults to None.
        population_size (int, optional):
            The number of individuals in the population (for population-based methods).
            BBOB recommendation: 10*dim for population-based algorithms.
            Defaults to 100.
        track_history (bool, optional):
            Whether to track optimization history for visualization and COCO postprocessing.
            When enabled, stores convergence data for performance analysis.
            Defaults to False.

    Attributes:
        func (Callable[[ndarray], float]):
            The objective function to be optimized.
        lower_bound (float):
            The lower bound of the search space.
        upper_bound (float):
            The upper bound of the search space.
        dim (int):
            The dimensionality of the search space.
        max_iter (int):
            The maximum number of iterations for the optimization process.
        seed (int):
            **REQUIRED for BBOB compliance.** The seed for the random number generator.
            Used for all random operations to ensure reproducibility.
        population_size (int):
            The number of individuals in the population.
        track_history (bool):
            Whether to track optimization history.
        history (dict[str, list]):
            Dictionary containing optimization history if track_history is True.
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

    @abstractmethod
    def search(self) -> tuple[ndarray, float]:
        """Perform the optimization search.

        Returns:
            Tuple containing the best solution found and its corresponding fitness value.
        """
