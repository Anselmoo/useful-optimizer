"""Cross-Entropy Method (CEM) optimizer implementation.

This module provides an implementation of the Cross-Entropy Method (CEM) optimizer. The
CEM algorithm is a stochastic optimization method that is particularly effective for
solving problems with continuous search spaces.

The CrossEntropyMethod class is the main class of this module and serves as the
optimizer. It takes an objective function, lower and upper bounds of the search space,
dimensionality of the search space, and other optional parameters as input. It uses the
CEM algorithm to find the optimal solution for the given objective function within the
specified search space.

Example usage:
    optimizer = CrossEntropyMethod(
        func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    elite_frac (float): The fraction of elite samples to select.
    noise_decay (float): The decay rate for the noise.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class CrossEntropyMethod(AbstractOptimizer):
    """Cross-Entropy Method optimizer.

    This optimizer uses the Cross-Entropy Method algorithm to find the optimal solution
    for a given function within a specified search space.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        elite_frac (float, optional): The fraction of elite samples to select. Defaults to 0.2.
        noise_decay (float, optional): The decay rate for the noise. Defaults to 0.99.
        seed (int | None, optional): The seed value for random number generation. Defaults to None.

    Attributes:
        elite_frac (float): The fraction of elite samples to select.
        noise_decay (float): The decay rate for the noise.

    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 100,
        max_iter: int = 1000,
        elite_frac: float = 0.2,
        noise_decay: float = 0.99,
        seed: int | None = None,
    ) -> None:
        """Initialize the CrossEntropyMethod class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.elite_frac = elite_frac
        self.noise_decay = noise_decay

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the search using the Cross-Entropy Method algorithm.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best sample found and its fitness value.

        """
        mean = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        std = np.ones(self.dim)

        for _ in range(self.max_iter):
            self.seed += 1
            samples = np.random.default_rng(self.seed).normal(
                mean, std, (self.population_size, self.dim)
            )
            samples = np.clip(samples, self.lower_bound, self.upper_bound)
            scores = np.array([self.func(sample) for sample in samples])
            elite_inds = scores.argsort()[: int(self.population_size * self.elite_frac)]
            elite_samples = samples[elite_inds]
            mean, std = elite_samples.mean(axis=0), elite_samples.std(axis=0)
            std *= self.noise_decay

        best_sample = mean
        best_fitness = self.func(best_sample)
        return best_sample, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(CrossEntropyMethod)
