"""Eagle Strategy Optimization Algorithm.

This module implements the Eagle Strategy (ES) optimization algorithm. ES is a
metaheuristic optimization algorithm inspired by the hunting behavior of eagles.
The algorithm mimics the way eagles soar, glide, and swoop down to catch their prey.

In ES, each eagle represents a potential solution, and the objective function
determines the quality of the solutions. The eagles try to update their positions by
mimicking the hunting behavior of eagles, which includes soaring, gliding, and swooping.

ES has been used for various kinds of optimization problems including function
optimization, neural network training, and other areas of engineering.

Example:
    optimizer = EagleStrategy(func=objective_function, lower_bound=-10, upper_bound=10,
    dim=2, n_eagles=50, max_iter=1000)
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    func (Callable): The objective function to optimize.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimension of the search space.
    n_eagles (int): The number of eagles (candidate solutions).
    max_iter (int): The maximum number of iterations.

Methods:
    search(): Perform the ES optimization.
"""

from __future__ import annotations

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


class EagleStrategy(AbstractOptimizer):
    """Implementation of the Eagle Strategy optimization algorithm.

    This class inherits from the AbstractOptimizer class and provides the search method
    to perform the optimization.

    Attributes:
        seed (int): The seed value for the random number generator.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        population_size (int): The size of the population.
        dim (int): The dimensionality of the problem.
        max_iter (int): The maximum number of iterations.

    Methods:
        search(): Performs the optimization and returns the best solution and its fitness value.


    Example:
        >>> from opt.metaheuristic.eagle_strategy import EagleStrategy
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = EagleStrategy(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = EagleStrategy(
        ...     func=shifted_ackley, dim=2,
        ...     lower_bound=-2.768, upper_bound=2.768,
        ...     max_iter=10, seed=42
        ... )
        >>> _, fitness = optimizer.search()
        >>> isinstance(float(fitness), float)
        True
    """

    def search(self) -> tuple[np.ndarray, float]:
        """Performs the optimization using the Eagle Strategy algorithm.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        # Initialize population and fitness
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(self.func, 1, population)

        # Initialize best solution
        best_index = np.argmin(fitness)
        best_solution = population[best_index]

        # Main loop
        for _ in range(self.max_iter):
            self.seed += 1
            for i in range(self.population_size):
                self.seed += 1
                # Generate a random solution for comparison
                random_solution = np.random.default_rng(self.seed).uniform(
                    self.lower_bound, self.upper_bound, self.dim
                )

                # If the random solution is better, move towards it
                if self.func(random_solution) < fitness[i]:
                    population[i] += np.random.default_rng(self.seed + 1).random() * (
                        random_solution - population[i]
                    )

                # Otherwise, move towards the best solution
                else:
                    population[i] += np.random.default_rng(self.seed + 2).random() * (
                        best_solution - population[i]
                    )

                # Update fitness
                fitness[i] = self.func(population[i])

                # Update best solution
                if fitness[i] < self.func(best_solution):
                    best_solution = population[i]

        return best_solution, self.func(best_solution)


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(EagleStrategy)
