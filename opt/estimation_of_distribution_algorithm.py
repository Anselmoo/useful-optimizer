"""Estimation of Distribution Algorithm optimizer.

This module implements the Estimation of Distribution Algorithm (EDA) optimizer.
The EDA optimizer is a population-based optimization algorithm that uses a probabilistic model
to estimate the distribution of promising solutions. It iteratively generates new solutions
by sampling from the estimated distribution.

The EstimationOfDistributionAlgorithm class is a subclass of the AbstractOptimizer class
and provides the implementation of the EDA optimizer. It initializes a population, selects
the best individuals based on fitness, estimates the mean and standard deviation of the
selected individuals, and generates new individuals by sampling from the estimated model.
The process is repeated for a specified number of iterations.

Example:
    To use the EstimationOfDistributionAlgorithm optimizer, create an instance of the class
    and call the search() method:

    ```python
    optimizer = EstimationOfDistributionAlgorithm(
        func=shifted_ackley,
        lower_bound=-32.768,
        upper_bound=+32.768,
        dim=2,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
    ```

Attributes:
    population_size (int): The size of the population.
    dim (int): The dimensionality of the problem.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    seed (int): The seed for the random number generator.
    max_iter (int): The maximum number of iterations.

Methods:
    _initialize(): Initializes the population.
    _select(population, fitness): Selects the best individuals based on fitness.
    _model(population): Estimates the mean and standard deviation of the selected individuals.
    _sample(mean, std): Generates new individuals by sampling from the estimated model.
    search(): Executes the search process and returns the best solution and fitness.
"""

from __future__ import annotations

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


class EstimationOfDistributionAlgorithm(AbstractOptimizer):
    """Estimation of Distribution Algorithm optimizer.

    This optimizer uses the Estimation of Distribution Algorithm (EDA) to search for
    the best solution. It initializes a population, selects the best individuals based
    on fitness, estimates the mean and standard deviation of the selected individuals,
    and generates new individuals by sampling from the estimated model.
    The process is repeated for a specified number of iterations.

    Args:
        AbstractOptimizer (class): The base class for all optimizers.

    Attributes:
        population_size (int): The size of the population.
        dim (int): The dimensionality of the problem.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        seed (int): The seed for the random number generator.
        max_iter (int): The maximum number of iterations.

    Methods:
        _initialize(): Initializes the population.
        _select(population, fitness): Selects the best individuals based on fitness.
        _model(population): Estimates the mean and standard deviation of the selected individuals.
        _sample(mean, std): Generates new individuals by sampling from the estimated model.
        search(): Executes the search process and returns the best solution and fitness.
    """

    def _initialize(self) -> np.ndarray:
        """Initialize the population.

        Returns:
            np.ndarray: The initialized population.
        """
        return np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

    def _select(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Select the best individuals based on fitness.

        Args:
            population (np.ndarray): The population of individuals.
            fitness (np.ndarray): The fitness values of the individuals.

        Returns:
            np.ndarray: The selected individuals.
        """
        idx = np.argsort(fitness)[: self.population_size // 2]
        return population[idx]

    def _model(self, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate the mean and standard deviation of the selected individuals.

        Args:
            population (np.ndarray): The selected individuals.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the estimated mean and standard deviation.

        """
        mean = np.mean(population, axis=0)
        std = np.std(population, axis=0)
        return mean, std

    def _sample(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Generate new individuals by sampling from the estimated model.

        Args:
            mean (np.ndarray): The estimated mean.
            std (np.ndarray): The estimated standard deviation.

        Returns:
            np.ndarray: The generated new individuals.

        """
        return np.random.default_rng(self.seed).normal(
            mean, std, (self.population_size, self.dim)
        )

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the search process and return the best solution and fitness.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution and its fitness.

        """
        population = self._initialize()
        best_solution: np.ndarray = np.empty(self.dim)
        best_fitness = np.inf
        for _ in range(self.max_iter):
            fitness = np.apply_along_axis(self.func, 1, population)
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_fitness:
                best_fitness = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]
            population = self._select(population, fitness)
            mean, std = self._model(population)
            population = self._sample(mean, std)
        return best_solution, best_fitness


if __name__ == "__main__":
    optimizer = EstimationOfDistributionAlgorithm(
        func=shifted_ackley, lower_bound=-32.768, upper_bound=+32.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
