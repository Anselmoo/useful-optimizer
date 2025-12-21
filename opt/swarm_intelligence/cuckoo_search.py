"""Cuckoo Search Optimization Algorithm.

This module implements the Cuckoo Search (CS) optimization algorithm.
CS is a nature-inspired metaheuristic algorithm, which is based on the obligate brood
parasitism of some cuckoo species. In these species, the cuckoos lay their eggs in the
nests of other host birds. If the host bird discovers the eggs are not their own, it
will either throw these alien eggs away or abandon its nest and build a completely new
one.

In the context of the CS algorithm, each egg in a nest represents a solution, and a
cuckoo egg represents a new solution. The aim is to use the new and potentially better
solutions (cuckoo eggs) to replace a not-so-good solution in the nests. In the simplest
form, each nest represents a solution, and thus the egg represents a new solution that
is to replace the old one if the new solution is better.

The CS algorithm is used to solve optimization problems by iteratively trying to
improve a candidate solution with regard to a given measure of quality, or fitness
function.

Example:
    optimizer = CuckooSearch(func=objective_function, lower_bound=-10, upper_bound=10,
    dim=2, n_nests=25, max_iter=1000)
    best_solution, best_fitness = optimizer.search()

Attributes:
    func (Callable): The objective function to optimize.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimension of the search space.
    n_nests (int): The number of nests (candidate solutions).
    max_iter (int): The maximum number of iterations.

Methods:
    search(): Perform the cuckoo search optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class CuckooSearch(AbstractOptimizer):
    """Cuckoo Search optimizer.

    This class implements the Cuckoo Search algorithm for optimization problems.

    Parameters:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The number of cuckoos in the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        mutation_probability (float, optional): The probability of mutation. Defaults to 0.1.
        seed (Optional[int], optional): The seed for the random number generator. Defaults to None.


    Example:
        >>> from opt.swarm_intelligence.cuckoo_search import CuckooSearch
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = CuckooSearch(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = CuckooSearch(
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
        mutation_probability: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize the CuckooSearch class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.mutation_probability = mutation_probability

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Cuckoo Search algorithm.

        Returns:
            Tuple[np.ndarray, float]: The best solution found and its corresponding fitness value.
        """
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(self.func, 1, population)

        for _ in range(self.max_iter):
            self.seed += 1
            for i in range(self.population_size):
                j = np.random.default_rng(self.seed + 1).choice(
                    [k for k in range(self.population_size) if k != i]
                )
                new_solution = population[i] + np.random.default_rng(
                    self.seed + 2
                ).normal(0, 1, self.dim) * (population[i] - population[j])
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                f_new = self.func(new_solution)
                if f_new < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = f_new

                elif (
                    np.random.default_rng(self.seed + 2).random()
                    < self.mutation_probability
                ):
                    new_solution = np.random.default_rng(self.seed + 3).uniform(
                        self.lower_bound, self.upper_bound, self.dim
                    )
                    f_new = self.func(new_solution)
                    if f_new < fitness[i]:
                        population[i] = new_solution
                        fitness[i] = f_new

        best_index = fitness.argmin()
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(CuckooSearch)
