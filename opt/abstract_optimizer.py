"""This module defines an abstract base class for optimizers."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class AbstractOptimizer(ABC):
    """An abstract base class for optimizers.

    Args:
        func (Callable): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations for the optimization process. Defaults to 1000.
        track_history (bool, optional): Whether to track optimization history for visualization. Defaults to False.

    Attributes:
        func (Callable): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int): The maximum number of iterations for the optimization process.
        seed (int): The seed for the random number generator.
        population_size (int): The number of individuals in the population.
        track_history (bool): Whether to track optimization history.
        history (dict): Dictionary containing optimization history if track_history is True.


    Methods:
        search() -> Tuple[TBestSolution, TBestFitness]:
            Perform the optimization search.

    Returns:
        Tuple containing the best solution found and its corresponding fitness value.
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 100,
        track_history: bool = False,
    ) -> None:
        """Initialize the optimizer."""
        self.func = func
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
    def search(self) -> tuple[np.ndarray, float]:
        """Perform the optimization search.

        Returns:
            Tuple containing the best solution found and its corresponding fitness value.
        """
