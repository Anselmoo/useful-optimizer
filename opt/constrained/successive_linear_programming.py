"""Successive Linear Programming optimization algorithm.

!!! warning

    This module is still under development and is not yet ready for use.

This module implements the Successive Linear Programming optimization algorithm. The
algorithm performs a search for the optimal solution by iteratively updating a
population of individuals. At each iteration, it computes the gradient of the objective
function for each individual and uses linear programming to find a new solution that
improves the objective function value. The process continues until the maximum number
of iterations is reached.

The SuccessiveLinearProgramming class is the main class that implements the algorithm.
It inherits from the AbstractOptimizer class and overrides the search() and gradient()
methods.

Attributes:
    seed (int): The seed value for the random number generator.
    lower_bound (float): The lower bound for the search space.
    upper_bound (float): The upper bound for the search space.
    population_size (int): The size of the population.
    dim (int): The dimensionality of the search space.
    max_iter (int): The maximum number of iterations.

Example usage:
    optimizer = SuccessiveLinearProgramming(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
"""

from __future__ import annotations

import numpy as np

from scipy.optimize import linprog

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


class SuccessiveLinearProgramming(AbstractOptimizer):
    """Implements the Successive Linear Programming optimization algorithm.

    This algorithm performs a search for the optimal solution by iteratively updating
    a population of individuals. At each iteration, it computes the gradient of the
    objective function for each individual and uses linear programming to find a new
    solution that improves the objective function value. The process continues until
    the maximum number of iterations is reached.

    Attributes:
        seed (int): The seed value for the random number generator.
        lower_bound (float): The lower bound for the search space.
        upper_bound (float): The upper bound for the search space.
        population_size (int): The size of the population.
        dim (int): The dimensionality of the search space.
        max_iter (int): The maximum number of iterations.


    Example:
        >>> from opt.constrained.successive_linear_programming import SuccessiveLinearProgramming
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SuccessiveLinearProgramming(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SuccessiveLinearProgramming(
        ...     func=shifted_ackley,
        ...     dim=2,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     max_iter=10,
        ...     seed=42,
        ... )
        >>> _, fitness = optimizer.search()
        >>> isinstance(float(fitness), float)
        True
    """

    def search(self) -> tuple[np.ndarray, float]:
        """Performs the search for the optimal solution.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found and
                its corresponding objective function value.
        """
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                gradient = self.gradient(population[i])
                bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
                result = linprog(c=gradient, bounds=bounds, method="highs")
                if result.success:
                    population[i] = result.x
        best_index = np.argmin([self.func(individual) for individual in population])
        return population[best_index], self.func(population[best_index])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Computes the gradient of the objective function at a given point.

        Args:
            x (np.ndarray): The point at which to compute the gradient.

        Returns:
            np.ndarray: The gradient vector.
        """
        eps = 1e-5
        return np.array(
            [
                (self.func(x + eps * unit_vector) - self.func(x)) / eps
                for unit_vector in np.eye(self.dim)
            ]
        )


if __name__ == "__main__":
    optimizer = SuccessiveLinearProgramming(
        func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
