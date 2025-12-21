"""Shuffled Frog Leaping Algorithm (SFLA) optimizer implementation.

This module provides an implementation of the Shuffled Frog Leaping Algorithm (SFLA)
optimizer. The SFLA is a population-based optimization algorithm inspired by the
behavior of frogs in a pond. It is used to solve optimization problems by iteratively
improving a population of candidate solutions.

The algorithm works by maintaining a population of frogs, where each frog represents a
candidate solution. In each iteration, the frogs are shuffled and leaped towards the
mean position of the best frogs. This process helps explore the search space and
converge towards the optimal solution.

This module defines the `ShuffledFrogLeapingAlgorithm` class, which is responsible for
executing the optimization process. The class takes an objective function, lower and
upper bounds of the search space, dimensionality of the search space, population size,
maximum number of iterations, and other optional parameters as input.

Example usage:
    optimizer = ShuffledFrogLeapingAlgorithm(
        func=shifted_ackley, lower_bound=-32.768, upper_bound=+32.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Fitness value: {best_fitness}")

Attributes:
    cut (int): The number of frogs to be used for leaping.
    seed (int | None): The seed for the random number generator.

Methods:
    __init__(self, func, lower_bound, upper_bound, dim, population_size=100,
        max_iter=1000, cut=2, seed=None)
        Initialize the ShuffledFrogLeapingAlgorithm class.

    search(self)
        Run the Shuffled Frog Leaping Algorithm (SFLA) optimization process.

Returns:
    tuple[np.ndarray, float]: A tuple containing the best frog position and its fitness value.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ShuffledFrogLeapingAlgorithm(AbstractOptimizer):
    """Implementation of the Shuffled Frog Leaping Algorithm (SFLA) optimizer.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        cut (int, optional): The number of frogs to be used for leaping. Defaults to 2.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Attributes:
        cut (int): The number of frogs to be used for leaping.
        seed (int | None): The seed for the random number generator.



    Example:
        >>> from opt.metaheuristic.shuffled_frog_leaping_algorithm import ShuffledFrogLeapingAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ShuffledFrogLeapingAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ShuffledFrogLeapingAlgorithm(
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
        cut: int = 2,
        seed: int | None = None,
    ) -> None:
        """Initialize the ShuffledFrogLeapingAlgorithm class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.cut = cut

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Shuffled Frog Leaping Algorithm (SFLA) optimization process.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best frog position and its fitness value.

        """
        # Initialize frog positions and fitness
        frogs = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.full(self.population_size, np.inf)
        best_frog: np.ndarray = np.array([])
        best_fitness = np.inf

        # Shuffled Frog Leaping Algorithm
        for _ in range(self.max_iter):
            self.seed += 1
            for i in range(self.population_size):
                self.seed += 1
                fitness[i] = self.func(frogs[i])

                # Update the best solution
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_frog = frogs[i].copy()

            # Shuffle the frogs
            order = np.argsort(fitness)
            frogs = frogs[order]

            # Leap the worst frog towards the mean position of the best frogs
            best_frogs = frogs[: self.population_size // self.cut]  # Use half the frogs
            mean_best_frog = np.mean(best_frogs, axis=0)
            frogs[-1] = frogs[-1] + np.random.default_rng(self.seed).random() * (
                mean_best_frog - frogs[-1]
            )

            # Ensure the frog positions stay within the bounds
            frogs = np.clip(frogs, self.lower_bound, self.upper_bound)

        return best_frog, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ShuffledFrogLeapingAlgorithm)
