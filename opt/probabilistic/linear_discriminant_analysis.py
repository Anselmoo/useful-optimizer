"""Linear Discriminant Analysis (LDA).

This module implements the Linear Discriminant Analysis (LDA). LDA is a method used in
statistics, pattern recognition, and machine learning to find a linear combination of
features that characterizes or separates two or more classes of objects or events.
The resulting combination may be used as a linear classifier, or, more commonly, for
dimensionality reduction before later classification.

LDA is closely related to analysis of variance (ANOVA) and regression analysis, which
also attempt to express one dependent variable as a linear combination of other
features or measurements. However, ANOVA uses categorical independent variables and a
continuous dependent variable, whereas discriminant analysis has continuous independent
variables and a categorical dependent variable (i.e., the class label).

Example:
    lda = LinearDiscriminantAnalysis(data, target)
    lda.fit()
    transformed_data = lda.transform()

Attributes:
    data (numpy.ndarray): The input data.
    target (numpy.ndarray): The class labels for the input data.

Methods:
    fit(): Fit the LDA model to the data.
    transform(): Apply the dimensionality reduction on the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import KBinsDiscretizer

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class LDAnalysis(AbstractOptimizer):
    """Linear Discriminant Analysis (LDA) optimizer.

    This optimizer uses Linear Discriminant Analysis (LDA) to perform search optimization.
    It searches for the best solution within a given search space by iteratively updating the population
    and fitness values based on the LDA predictions.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        number_of_labels (int, optional): The number of labels for fitness discretization. Defaults to 20.
        unique_classes (int, optional): The minimum number of unique classes in the fitness values. Defaults to 2.
        seed (Optional[int], optional): The seed value for random number generation. Defaults to None.

    Attributes:
        population (np.ndarray): The current population array.
        fitness (np.ndarray): The current fitness array.
        lda (LinearDiscriminantAnalysis): The instance of LinearDiscriminantAnalysis used for prediction.
        discretizer (KBinsDiscretizer): The instance of KBinsDiscretizer used for fitness discretization.

    Methods:
        _make_lda(): Create an instance of LinearDiscriminantAnalysis.
        vectorize(population, fitness): Vectorize the population and fitness values.
        search(): Perform the search optimization.



    Example:
        >>> from opt.probabilistic.linear_discriminant_analysis import LDAnalysis
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = LDAnalysis(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = LDAnalysis(
        ...     func=shifted_ackley, dim=2,
        ...     lower_bound=-2.768, upper_bound=2.768,
        ...     max_iter=10, seed=42
        ... )
        >>> _, fitness = optimizer.search()
        >>> isinstance(float(fitness), float)
        True
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 100,
        max_iter: int = 1000,
        number_of_labels: int = 20,
        unique_classes: int = 2,
        seed: int | None = None,
    ) -> None:
        """Initialize the LDAnalysis class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )

        self.population: np.ndarray = np.empty((self.population_size, self.dim))
        self.fitness = np.inf
        self.lda = self._make_lda()
        self.discretizer = KBinsDiscretizer(
            n_bins=number_of_labels, encode="ordinal", strategy="uniform"
        )
        self.minum_unique_classes = unique_classes

    def _make_lda(self) -> LinearDiscriminantAnalysis:
        """Create an instance of LinearDiscriminantAnalysis.

        Returns:
            LinearDiscriminantAnalysis: An instance of LinearDiscriminantAnalysis.
        """
        return LinearDiscriminantAnalysis(solver="lsqr")

    def vectorize(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorize the population and fitness values.

        Args:
            population (np.ndarray): The population array.
            fitness (np.ndarray): The fitness array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The vectorized population and fitness arrays.
        """
        return population.reshape(-1, self.dim), fitness

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the search optimization.

        Returns:
            Tuple[np.ndarray, float]: The best solution found and its fitness value.
        """
        self.population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.population)

        # Discretize the fitness values
        self.fitness = self.discretizer.fit_transform(
            self.fitness.reshape(-1, 1)
        ).ravel()

        for _ in range(self.max_iter):
            x, y = self.vectorize(self.population, self.fitness)
            x, y = np.nan_to_num(x), np.nan_to_num(y)

            # Check if there are at least two unique classes in y
            if len(np.unique(y)) < self.minum_unique_classes:
                continue

            self.lda.fit(x, y)

            def _upper_bound(x: np.ndarray) -> float:
                """Upper bound function."""
                x = x.reshape(1, -1)
                return self.lda.predict(x)

            optimal_val = np.inf
            optimal_x: np.ndarray = np.empty(self.dim)
            num_restarts = 50

            x_seeds = np.random.default_rng(self.seed).uniform(
                self.lower_bound, self.upper_bound, (num_restarts, self.dim)
            )

            for x0 in x_seeds:
                res = minimize(
                    _upper_bound,
                    x0,
                    bounds=[(self.lower_bound, self.upper_bound)] * self.dim,
                    method="L-BFGS-B",
                )
                if res.fun < optimal_val:
                    optimal_val = res.fun
                    optimal_x = res.x

            # Update the population and fitness
            self.population = np.vstack([self.population, optimal_x])
            new_fitness = self.func(optimal_x)
            new_fitness = self.discretizer.transform(
                new_fitness.reshape(-1, 1)
            ).ravel()[0]
            self.fitness = np.append(self.fitness, new_fitness)

        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(LDAnalysis)
