"""Imperialist Competitive Algorithm optimizer.

This module implements the Imperialist Competitive Algorithm (ICA) for solving
optimization problems. The ICA is a population-based algorithm that simulates the
competition between empires and colonies. It starts with a random population and
iteratively improves the solutions by assimilation, revolution, position exchange,
and imperialistic competition.

Example:
    To use this optimizer, create an instance of the `ImperialistCompetitiveAlgorithm` class and call
    the `search` method to run the optimization.

        from opt.imperialist_competitive_algorithm import ImperialistCompetitiveAlgorithm
        from opt.benchmark.functions import shifted_ackley

        # Define the objective function
        def objective_function(x):
            return shifted_ackley(x)

        # Create an instance of the optimizer
        optimizer = ImperialistCompetitiveAlgorithm(
            func=objective_function,
            dim=2,
            lower_bound=-32.768,
            upper_bound=32.768,
            num_empires=15,
            population_size=100,
            max_iter=1000,
        )

        # Run the optimization
        best_solution, best_fitness = optimizer.search()

        print(f"Best solution found: {best_solution}")
        print(f"Best fitness value: {best_fitness}")

Attributes:
    num_empires (int): The number of empires in the algorithm.
    revolution_rate (float): The rate of revolution, which determines the probability of a revolution occurring.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ImperialistCompetitiveAlgorithm(AbstractOptimizer):
    """Imperialist Competitive Algorithm optimizer.

    This optimizer implements the Imperialist Competitive Algorithm (ICA) for solving
    optimization problems. It is a population-based algorithm that simulates the
    competition between empires and colonies.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        dim (int): The dimensionality of the problem.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        num_empires (int, optional): The number of empires. Defaults to 15.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        revolution_rate (float, optional): The rate of revolution. Defaults to 0.3.
        seed (int | None, optional): The random seed. Defaults to None.

    Attributes:
        num_empires (int): The number of empires.
        revolution_rate (float): The rate of revolution.



    Example:
        >>> from opt.evolutionary.imperialist_competitive_algorithm import ImperialistCompetitiveAlgorithm
        >>> from opt.benchmark.functions import rosenbrock
        >>> optimizer = ImperialistCompetitiveAlgorithm(
        ...     func=rosenbrock, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ImperialistCompetitiveAlgorithm(
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
        num_empires: int = 15,
        population_size: int = 100,
        max_iter: int = 1000,
        revolution_rate: float = 0.3,
        seed: int | None = None,
    ) -> None:
        """Initialize the ImperialistCompetitiveAlgorithm class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )

        self.num_empires = num_empires
        self.revolution_rate = revolution_rate

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Imperialist Competitive Algorithm optimization.

        Returns:
            tuple[np.ndarray, float]: The best solution found and its fitness value.

        """
        # Initialize population
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(self.func, 1, population)

        # Create empires
        empires = []
        for i in range(self.num_empires):
            empire = {
                "imperialist": i,
                "colonies": list(range(self.num_empires, self.num_empires + i + 1)),
                "total_cost": fitness[i]
                + fitness[self.num_empires + i : self.num_empires + i + 1].sum(),
            }
            empires.append(empire)

        for _ in range(self.max_iter):
            self.seed += 1
            # Assimilation
            for empire in empires:
                self.seed += 1
                population[empire["colonies"]] += np.random.default_rng(
                    self.seed
                ).random((len(empire["colonies"]), self.dim)) * (
                    population[empire["imperialist"]] - population[empire["colonies"]]
                )

            # Revolution
            revolution_indices = (
                np.random.default_rng(self.seed).random(len(population))
                < self.revolution_rate
            )
            population[revolution_indices] = np.random.default_rng(self.seed).uniform(
                self.lower_bound, self.upper_bound, (revolution_indices.sum(), self.dim)
            )

            # Position exchange between a colony and Imperialist
            for empire in empires:
                self.seed += 1
                colonies_fitness = np.apply_along_axis(
                    self.func, 1, population[empire["colonies"]]
                )
                if colonies_fitness.min() < fitness[empire["imperialist"]]:
                    best_colony_index = colonies_fitness.argmin()
                    (
                        population[empire["imperialist"]],
                        population[empire["colonies"][best_colony_index]],
                    ) = (
                        population[empire["colonies"][best_colony_index]],
                        population[empire["imperialist"]],
                    )

            # Imperialistic competition
            total_power = sum([1 / empire["total_cost"] for empire in empires])
            for i in range(len(empires)):
                self.seed += 1
                for j in range(i + 1, len(empires)):
                    self.seed += 1
                    if (
                        np.random.default_rng(self.seed).random()
                        < (1 / empires[i]["total_cost"]) / total_power
                    ) and len(empires[j]["colonies"]) > 0:
                        lost_colony = empires[j]["colonies"].pop()
                        empires[i]["colonies"].append(lost_colony)

            # Eliminate the powerless empires
            empires = [empire for empire in empires if len(empire["colonies"]) > 0]

        best_solution = min(
            empires, key=lambda empire: self.func(population[empire["imperialist"]])
        )
        best_fitness = self.func(population[best_solution["imperialist"]])
        return population[best_solution["imperialist"]], best_fitness


if __name__ == "__main__":
    optimizer = ImperialistCompetitiveAlgorithm(
        func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
