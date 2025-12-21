"""Whale Optimization Algorithm (WOA).

This module implements the Whale Optimization Algorithm (WOA). WOA is a metaheuristic
optimization algorithm inspired by the hunting behavior of humpback whales.
The algorithm is based on the echolocation behavior of humpback whales, which use sounds
to communicate, navigate and hunt in dark or murky waters.

In WOA, each whale represents a potential solution, and the objective function
determines the quality of the solutions. The whales try to update their positions by
mimicking the hunting behavior of humpback whales, which includes encircling,
bubble-net attacking, and searching for prey.

WOA has been used for various kinds of optimization problems including function
optimization, neural network training, and other areas of engineering.

Example:
    optimizer = WhaleOptimizationAlgorithm(func=objective_function, lower_bound=-10,
    upper_bound=10, dim=2, n_whales=50, max_iter=1000)
    best_solution, best_fitness = optimizer.search()

Attributes:
    func (Callable): The objective function to optimize.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimension of the search space.
    n_whales (int): The number of whales (candidate solutions).
    max_iter (int): The maximum number of iterations.

Methods:
    search(): Perform the WOA optimization.
"""

from __future__ import annotations

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


class WhaleOptimizationAlgorithm(AbstractOptimizer):
    """Implementation of the Whale Optimization Algorithm (WOA) for optimization problems.

    WOA is a metaheuristic optimization algorithm inspired by the hunting behavior of
    humpback whales. It iteratively updates the positions of a population of whales to
    search for the optimal solution.

    Attributes:
        population_size (int): The number of whales in the population.
        dim (int): The dimensionality of the problem.
        max_iter (int): The maximum number of iterations.
        lower_bound (float or array-like): The lower bound(s) of the search space.
        upper_bound (float or array-like): The upper bound(s) of the search space.
        seed (int): The random seed for reproducibility.
        func (callable): The objective function to be minimized.

    Methods:
        search(): Runs the Whale Optimization Algorithm and returns the best solution found.



    Example:
        >>> from opt.swarm_intelligence.whale_optimization_algorithm import (
        ...     WhaleOptimizationAlgorithm,
        ... )
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = WhaleOptimizationAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = WhaleOptimizationAlgorithm(
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
        """Runs the Whale Optimization Algorithm and returns the best solution found.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best solution found (as a numpy array)
            and its corresponding fitness value (a float).

        """
        # Initialize whale positions and fitness
        whales = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.full(self.population_size, np.inf)
        best_whale: np.ndarray = np.empty(self.dim)
        best_fitness = np.inf
        fifty = 0.5

        # Whale Optimization Algorithm
        for iter_count in range(self.max_iter):
            self.seed += 1
            for i in range(self.population_size):
                fitness[i] = self.func(whales[i])

                # Update the best solution
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_whale = whales[i].copy()

            a = 2 - iter_count * ((2) / self.max_iter)  # Linearly decreasing a

            for i in range(self.population_size):
                self.seed += 1
                r1 = np.random.default_rng(
                    self.seed + 1
                ).random()  # r1 is a random number in [0,1]
                r2 = np.random.default_rng(
                    self.seed + 2
                ).random()  # r2 is a random number in [0,1]

                A = 2 * a * r1 - a
                C = 2 * r2

                b = 1  # parameters in equation (2.3)
                l = (a - 1) * np.random.default_rng(
                    self.seed + 3
                ).random() + 1  # parameters in equation (2.3)

                p = np.random.default_rng(self.seed + 4).random()  # p in equation (2.6)

                for j in range(self.dim):
                    self.seed += 1
                    if p < fifty:
                        if abs(A) >= 1:
                            rand_leader_index = np.random.default_rng(
                                self.seed
                            ).integers(0, self.population_size)
                            X_rand = whales[rand_leader_index]
                            whales[i][j] = X_rand[j] - A * abs(
                                C * X_rand[j] - whales[i][j]
                            )
                        elif abs(A) < 1:
                            whales[i][j] = best_whale[j] - A * abs(
                                C * best_whale[j] - whales[i][j]
                            )
                    elif p >= fifty:
                        distance2Leader = abs(best_whale[j] - whales[i][j])
                        whales[i][j] = (
                            distance2Leader * np.exp(b * l) * np.cos(l * 2 * np.pi)
                            + best_whale[j]
                        )

                whales[i] = np.clip(whales[i], self.lower_bound, self.upper_bound)

        return best_whale, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(WhaleOptimizationAlgorithm)
