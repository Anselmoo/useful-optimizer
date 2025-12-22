"""Arithmetic Optimization Algorithm (AOA) implementation.

This module implements the Arithmetic Optimization Algorithm, a math-inspired
metaheuristic optimization algorithm based on arithmetic operators.

Reference:
    Abualigah, L., Diabat, A., Mirjalili, S., Abd Elaziz, M., & Gandomi, A. H.
    (2021). The arithmetic optimization algorithm. Computer Methods in Applied
    Mechanics and Engineering, 376, 113609.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_ALPHA = 5.0  # Sensitivity parameter for exploitation
_MU = 0.5  # Control parameter for search
_MIN_VALUE = 1e-10  # Minimum value to avoid division by zero


class ArithmeticOptimizationAlgorithm(AbstractOptimizer):
    """Arithmetic Optimization Algorithm optimizer.

    The AOA uses basic arithmetic operations to explore and exploit:
    - Multiplication and Division for exploration
    - Subtraction and Addition for exploitation

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of solutions in the population.


    Example:
        >>> from opt.metaheuristic.arithmetic_optimization import ArithmeticOptimizationAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ArithmeticOptimizationAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ArithmeticOptimizationAlgorithm(
        ...     func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=2.768, max_iter=10
        ... )
        >>> _, fitness = optimizer.search()
        >>> isinstance(float(fitness), float)
        True
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        population_size: int = 30,
    ) -> None:
        """Initialize the Arithmetic Optimization Algorithm.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of solutions in the population.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Arithmetic Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Find best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Main loop
        for iteration in range(self.max_iter):
            # Calculate Math Optimizer Accelerated (MOA) function
            moa = 0.2 + (1 - iteration / self.max_iter) ** (_ALPHA)

            # Calculate Math Optimizer Probability (MOP) function
            mop = 1 - ((iteration) ** (1 / _ALPHA)) / ((self.max_iter) ** (1 / _ALPHA))

            for i in range(self.population_size):
                new_position = np.zeros(self.dim)

                for j in range(self.dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    r3 = np.random.rand()

                    if r1 > moa:
                        # Exploration phase (Multiplication or Division)
                        if r2 > 0.5:
                            # Division
                            divisor = mop * (
                                (self.upper_bound - self.lower_bound) * _MU
                                + self.lower_bound
                            )
                            if abs(divisor) < _MIN_VALUE:
                                divisor = _MIN_VALUE
                            new_position[j] = best_solution[j] / divisor
                        else:
                            # Multiplication
                            new_position[j] = (
                                best_solution[j]
                                * mop
                                * (
                                    (self.upper_bound - self.lower_bound) * _MU
                                    + self.lower_bound
                                )
                            )
                    # Exploitation phase (Subtraction or Addition)
                    elif r3 > 0.5:
                        # Subtraction
                        new_position[j] = best_solution[j] - mop * (
                            (self.upper_bound - self.lower_bound) * _MU
                            + self.lower_bound
                        )
                    else:
                        # Addition
                        new_position[j] = best_solution[j] + mop * (
                            (self.upper_bound - self.lower_bound) * _MU
                            + self.lower_bound
                        )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new solution
                new_fitness = self.func(new_position)

                # Greedy selection
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ArithmeticOptimizationAlgorithm)
