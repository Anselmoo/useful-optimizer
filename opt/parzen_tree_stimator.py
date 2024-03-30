"""Parzen Tree Estimator optimizer.

The Parzen Tree Estimator optimizer is an algorithm that uses the Parzen Tree Estimator
technique to search for the optimal solution of a given function within a specified
search space. It is particularly useful for optimization problems where the objective
function is expensive to evaluate.

The Parzen Tree Estimator algorithm works by maintaining a population of
hyperparameters and their corresponding scores. It segments the population into two
distributions based on the scores and fits Gaussian kernel density estimators to each
distribution. It then samples hyperparameters from the low score distribution and
selects the hyperparameters with the highest score difference or ratio between the
low and high score distributions. This process is iteratively repeated to search
for the optimal solution.

This implementation of the Parzen Tree Estimator optimizer provides a flexible and
customizable framework for solving optimization problems. It allows users to specify
the objective function, search space, population size, maximum number of iterations,
selection strategy, and other parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sklearn.neighbors import KernelDensity

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ParzenTreeEstimator(AbstractOptimizer):
    """Parzen Tree Estimator optimizer.

    This optimizer uses the Parzen Tree Estimator algorithm to search for the optimal solution
    of a given function within a specified search space.

    Args:
        func (Callable): The objective function to be optimized.
        dim (int): The dimensionality of the search space.
        lower_bound (float or array-like): The lower bound(s) of the search space.
        upper_bound (float or array-like): The upper bound(s) of the search space.
        population_size (int, optional): The size of the population. Default is 100.
        max_iter (int, optional): The maximum number of iterations. Default is 1000.
        gamma (float, optional): The quantile value used to segment the distributions. Default is 0.15.
        bandwidth (float, optional): The bandwidth of the Gaussian kernel used in the Parzen Tree Estimator. Default is 0.2.
        n_samples (int, optional): The number of samples to draw from the estimated distributions. Default is None.
        selection_strategy (str, optional): The selection strategy used to choose the next set of hyperparameters.
            - "difference": Selects the hyperparameters with the largest difference in scores between the two distributions.
            - "ratio": Selects the hyperparameters with the largest ratio of scores between the two distributions.
            Default is "difference".
        seed (int, optional): The seed value for the random number generator. Default is None.

    Attributes:
        gamma (float): The quantile value used to segment the distributions.
        bandwidth (float): The bandwidth of the Gaussian kernel used in the Parzen Tree Estimator.
        n_samples (int): The number of samples to draw from the estimated distributions.
        population (ndarray): The current population of hyperparameters.
        scores (ndarray): The scores of the hyperparameters in the population.
        sample_select (Callable): The selection function based on the selection strategy.

    Methods:
        initialize_population(): Initializes the population of hyperparameters.
        segment_distributions(): Segments the distributions based on the scores.
        choose_next_hps(l_kde, g_kde): Chooses the next set of hyperparameters based on the selection strategy.
        search(): Executes the Parzen Tree Estimator algorithm to find the optimal solution.
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        dim: int,
        lower_bound: float,
        upper_bound: float,
        population_size: int = 100,
        max_iter: int = 1000,
        gamma: float = 0.15,
        bandwidth: float = 0.2,
        n_samples: int | None = None,
        selection_strategy: str = "difference",
        seed: int | None = None,
    ) -> None:
        """Initialize the ParzenTreeEstimator class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.gamma = gamma
        self.bandwidth = bandwidth
        if n_samples is None:
            n_samples = population_size
        self.n_samples = n_samples
        self.population = np.empty((population_size, dim))
        self.scores = np.inf * np.ones(population_size)
        if selection_strategy == "difference":
            self.sample_select = lambda l_score, g_score: np.argmax(l_score - g_score)
        elif selection_strategy == "ratio":
            self.sample_select = lambda l_score, g_score: np.argmax(g_score / l_score)
        else:
            msg = f"Invalid selection strategy: {selection_strategy}"
            raise ValueError(msg)

    def initialize_population(self) -> None:
        """Initializes the population of hyperparameters.

        This method generates a random population of hyperparameters within the specified search space.
        """
        self.population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        self.scores = np.array(
            [self.func(individual) for individual in self.population]
        )

    def segment_distributions(self) -> tuple[KernelDensity, KernelDensity]:
        """Segments the distributions based on the scores.

        This method segments the population into two distributions based on the scores of the hyperparameters.
        It fits a Gaussian kernel density estimator to each distribution.

        Returns:
            Tuple[KernelDensity, KernelDensity]: The fitted kernel density estimators for the low and high score distributions.
        """
        cut = np.quantile(self.scores, self.gamma)
        l_x = self.population[self.scores < cut]
        g_x = self.population[self.scores >= cut]
        l_kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth).fit(l_x)
        g_kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth).fit(g_x)
        return l_kde, g_kde

    def choose_next_hps(self, l_kde: KernelDensity, g_kde: KernelDensity) -> np.ndarray:
        """Chooses the next set of hyperparameters based on the selection strategy.

        This method samples hyperparameters from the low score distribution and selects the hyperparameters
        with the highest score difference or ratio between the low and high score distributions.

        Parameters:
            l_kde (KernelDensity): The kernel density estimator for the low score distribution.
            g_kde (KernelDensity): The kernel density estimator for the high score distribution.

        Returns:
            np.ndarray: The selected set of hyperparameters.
        """
        samples = l_kde.sample(self.n_samples)
        l_score = l_kde.score_samples(samples)
        g_score = g_kde.score_samples(samples)
        return samples[self.sample_select(l_score, g_score)]

    def search(self) -> tuple[np.ndarray, float]:
        """Executes the Parzen Tree Estimator algorithm to find the optimal solution.

        This method iteratively performs the Parzen Tree Estimator algorithm to search for the optimal solution.
        It updates the population of hyperparameters based on the scores and selects the best solution.

        Returns:
            Tuple[np.ndarray, float]: The best set of hyperparameters and its corresponding score.
        """
        self.initialize_population()
        for _ in range(self.max_iter):
            self.seed += 1
            l_kde, g_kde = self.segment_distributions()
            hps = self.choose_next_hps(l_kde, g_kde)
            score = self.func(hps)
            worst_index = np.argmax(self.scores)
            self.population[worst_index] = hps
            self.scores[worst_index] = score
        best_index = np.argmin(self.scores)
        return self.population[best_index], self.scores[best_index]


if __name__ == "__main__":
    optimizer = ParzenTreeEstimator(
        func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
