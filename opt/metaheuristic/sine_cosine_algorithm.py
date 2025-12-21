"""Sine Cosine Algorithm optimization algorithm.

This module implements the Sine Cosine Algorithm (SCA) optimization algorithm.
SCA is a population-based metaheuristic algorithm inspired by the sine and cosine
functions. It is commonly used for solving optimization problems.

The SineCosineAlgorithm class provides an implementation of the SCA algorithm. It takes
an objective function, lower and upper bounds of the search space, dimensionality of
the search space, and other optional parameters as input. The search method performs
the optimization and returns the best solution found along with its fitness value.

Example:
    import numpy as np
    from opt.benchmark.functions import shifted_ackley

    # Create an instance of SineCosineAlgorithm optimizer
    optimizer = SineCosineAlgorithm(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=2000,
    )

    # Perform the optimization
    best_solution, best_fitness = optimizer.search()

    # Print the results
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

Attributes:
    r1_cut (float): The threshold for selecting the sine update rule.
    r2_cut (float): The threshold for selecting the cosine update rule.

Methods:
    search(): Perform the Sine Cosine Algorithm optimization.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class SineCosineAlgorithm(AbstractOptimizer):
    """The SineCosineAlgorithm class implements the Sine Cosine Algorithm optimization algorithm.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population (default: 100).
        max_iter (int, optional): The maximum number of iterations (default: 1000).
        r1_cut (float, optional): The threshold for selecting the sine update rule (default: 0.5).
        r2_cut (float, optional): The threshold for selecting the cosine update rule (default: 0.5).
        seed (int | None, optional): The seed value for random number generation (default: None).

    Attributes:
        r1_cut (float): The threshold for selecting the sine update rule.
        r2_cut (float): The threshold for selecting the cosine update rule.

    Methods:
        search(): Perform the Sine Cosine Algorithm optimization.



    Example:
        >>> from opt.metaheuristic.sine_cosine_algorithm import SineCosineAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SineCosineAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SineCosineAlgorithm(
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
        r1_cut: float = 0.5,
        r2_cut: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize the SineCosineAlgorithm class.

        Args:
            func (Callable[[ndarray], float]): The objective function to be optimized.
            lower_bound (float): The lower bound of the search space.
            upper_bound (float): The upper bound of the search space.
            dim (int): The dimensionality of the search space.
            population_size (int, optional): The size of the population (default: 100).
            max_iter (int, optional): The maximum number of iterations (default: 1000).
            r1_cut (float, optional): The threshold for selecting the sine update rule (default: 0.5).
            r2_cut (float, optional): The threshold for selecting the cosine update rule (default: 0.5).
            seed (int | None, optional): The seed value for random number generation (default: None).
        """
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.r1_cut = r1_cut
        self.r2_cut = r2_cut

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the Sine Cosine Algorithm optimization.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found and its corresponding fitness value.
        """
        # Initialize population and fitness
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(self.func, 1, population)

        # Main loop
        for _ in range(self.max_iter):
            self.seed += 1
            # Get best solution
            best_index = np.argmin(fitness)
            best_solution = population[best_index]

            for i in range(self.population_size):
                self.seed += 1
                # Update position
                for j in range(self.dim):
                    self.seed += 1
                    r1 = np.random.default_rng(
                        self.seed + 1
                    ).random()  # r1 is a random number in [0,1]
                    r2 = np.random.default_rng(
                        self.seed + 2
                    ).random()  # r2 is a random number in [0,1]

                    # Update position based on the Sine Cosine Algorithm update rule
                    if r1 < self.r1_cut:
                        if r2 < self.r2_cut:
                            population[i][j] += np.sin(
                                np.random.default_rng(self.seed + 4).random()
                            ) * abs(
                                np.random.default_rng(self.seed + 5).random()
                                * best_solution[j]
                                - population[i][j]
                            )
                        else:
                            population[i][j] += np.cos(
                                np.random.default_rng(self.seed + 6).random()
                            ) * abs(
                                np.random.default_rng(self.seed + 7).random()
                                * best_solution[j]
                                - population[i][j]
                            )
                    elif r2 < self.r2_cut:
                        population[i][j] -= np.sin(
                            np.random.default_rng(self.seed + 8).random()
                        ) * abs(
                            np.random.default_rng(self.seed + 9).random()
                            * best_solution[j]
                            - population[i][j]
                        )
                    else:
                        population[i][j] -= np.cos(
                            np.random.default_rng(self.seed + 10).random()
                        ) * abs(
                            np.random.default_rng(self.seed + 11).random()
                            * best_solution[j]
                            - population[i][j]
                        )

                # Ensure the position stays within the bounds
                population[i] = np.clip(
                    population[i], self.lower_bound, self.upper_bound
                )

                # Update fitness
                fitness[i] = self.func(population[i])

                # Update best solution
                if fitness[i] < fitness[best_index]:
                    best_index = i
                    best_solution = population[i]
        best_fitness = fitness[best_index]
        return best_solution, best_fitness


if __name__ == "__main__":
    optimizer = SineCosineAlgorithm(
        func=shifted_ackley, dim=2, lower_bound=-32.768, upper_bound=+32.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
