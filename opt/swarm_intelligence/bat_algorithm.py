"""Bat Algorithm optimization algorithm.

This module implements the Bat Algorithm optimization algorithm. The Bat Algorithm is a
metaheuristic algorithm inspired by the echolocation behavior of bats. It is commonly
used for solving optimization problems.

The BatAlgorithm class provides an implementation of the Bat Algorithm optimization
algorithm. It takes an objective function, the dimensionality of the problem, the
search space bounds, the number of bats in the population, and other optional
parameters. The search method runs the Bat Algorithm optimization and returns the
best solution found.

Example:
    import numpy as np
    from opt.benchmark.functions import shifted_ackley
    from opt.bat_algorithm import BatAlgorithm

    # Define the objective function
    def objective_function(x):
        return np.sum(x ** 2)

    # Create an instance of the BatAlgorithm class
    optimizer = BatAlgorithm(
        func=objective_function,
        dim=2,
        lower_bound=-5.0,
        upper_bound=5.0,
        n_bats=10,
        max_iter=1000,
        loudness=0.5,
        pulse_rate=0.9,
        freq_min=0,
        freq_max=2,
        seed=42
    )

    # Run the Bat Algorithm optimization
    best_solution, best_fitness = optimizer.search()

    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

Attributes:
    freq_min (float): The minimum frequency of the bats.
    freq_max (float): The maximum frequency of the bats.
    positions (ndarray): The current positions of the bats.
    velocities (ndarray): The velocities of the bats.
    frequencies (ndarray): The frequencies of the bats.
    loudnesses (ndarray): The loudnesses of the bats.
    best_positions (ndarray): The best positions found by each bat.
    best_fitnesses (ndarray): The fitness values corresponding to the best positions found by each bat.
    alpha (float): The pulse rate of the bats.
    gamma (float): The loudness of the bats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class BatAlgorithm(AbstractOptimizer):
    """Implementation of the Bat Algorithm optimization algorithm.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        dim (int): The dimensionality of the problem.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        n_bats (int): The number of bats in the population.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        loudness (float, optional): The initial loudness of the bats. Defaults to 0.5.
        pulse_rate (float, optional): The pulse rate of the bats. Defaults to 0.9.
        freq_min (float, optional): The minimum frequency of the bats. Defaults to 0.
        freq_max (float, optional): The maximum frequency of the bats. Defaults to 2.
        seed (int | None, optional): The seed value for random number generation. Defaults to None.

    Attributes:
        freq_min (float): The minimum frequency of the bats.
        freq_max (float): The maximum frequency of the bats.
        positions (ndarray): The current positions of the bats.
        velocities (ndarray): The velocities of the bats.
        frequencies (ndarray): The frequencies of the bats.
        loudnesses (ndarray): The loudnesses of the bats.
        best_positions (ndarray): The best positions found by each bat.
        best_fitnesses (ndarray): The fitness values corresponding to the best positions found by each bat.
        alpha (float): The pulse rate of the bats.
        gamma (float): The loudness of the bats.



    Example:
        >>> from opt.swarm_intelligence.bat_algorithm import BatAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = BatAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = BatAlgorithm(
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
        dim: int,
        lower_bound: float,
        upper_bound: float,
        n_bats: int,
        max_iter: int = 1000,
        loudness: float = 0.5,
        pulse_rate: float = 0.9,
        freq_min: float = 0,
        freq_max: float = 2,
        seed: int | None = None,
    ) -> None:
        """Initialize the BatAlgorithm class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=n_bats,
        )

        self.freq_min = freq_min
        self.freq_max = freq_max
        self.positions = np.random.default_rng(self.seed).uniform(
            lower_bound, upper_bound, (self.population_size, dim)
        )
        self.velocities = np.zeros((self.population_size, dim))
        self.frequencies = np.random.default_rng(self.seed).uniform(
            freq_min, freq_max, self.population_size
        )
        self.loudnesses = np.full(self.population_size, loudness)
        self.best_positions = self.positions.copy()
        self.best_fitnesses = np.full(self.population_size, np.inf)
        self.alpha = pulse_rate
        self.gamma = loudness

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Bat Algorithm optimization.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found (position) and its fitness value.

        """
        best_solution_idx = None
        for _ in range(self.max_iter):
            self.seed += 1
            for i in range(self.population_size):
                self.seed += 1
                self.positions[i] += self.velocities[i]
                fitness = self.func(self.positions[i])
                if fitness < self.best_fitnesses[i]:
                    self.best_positions[i] = self.positions[i].copy()
                    self.best_fitnesses[i] = fitness
                if (
                    best_solution_idx is None
                    or fitness < self.best_fitnesses[best_solution_idx]
                ):
                    best_solution_idx = i
                self.velocities[i] += (
                    self.best_positions[best_solution_idx] - self.positions[i]
                ) * self.loudnesses[i]
                self.frequencies[i] = (
                    self.freq_min
                    + (self.freq_max - self.freq_min)
                    * np.random.default_rng(self.seed).random()
                )
                if np.random.default_rng(self.seed + 1).random() > self.loudnesses[i]:
                    self.seed += 1
                    self.positions[i] = self.best_positions[
                        best_solution_idx
                    ] + self.alpha * np.random.default_rng(self.seed).normal(
                        0, 1, self.dim
                    )
            self.loudnesses *= self.gamma
        return self.best_positions[best_solution_idx], self.best_fitnesses[
            best_solution_idx
        ]


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(BatAlgorithm)
