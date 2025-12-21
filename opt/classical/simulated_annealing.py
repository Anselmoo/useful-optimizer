"""Simulated Annealing optimizer.

This module provides an implementation of the Simulated Annealing optimization
algorithm. Simulated Annealing is a metaheuristic optimization algorithm that is
inspired by the annealing process in metallurgy. It is used to find the global minimum
of a given objective function in a search space.

Example:
    To use the SimulatedAnnealing optimizer, create an instance of the class and call the `search` method:

    ```python
    optimizer = SimulatedAnnealing(func, lower_bound, upper_bound, dim)
    best_solution, best_cost = optimizer.search()
    ```

Classes:
    SimulatedAnnealing: A class that implements the Simulated Annealing optimization algorithm.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class SimulatedAnnealing(AbstractOptimizer):
    """Simulated Annealing optimizer.

    The SimulatedAnnealing class implements the Simulated Annealing optimization algorithm.
    It is used to find the global minimum of a given objective function in a search space.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The number of individuals in the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        init_temperature (float, optional): The initial temperature. Defaults to 1000.
        stopping_temperature (float, optional): The stopping temperature. Defaults to 1e-8.
        cooling_rate (float, optional): The cooling rate. Defaults to 0.99.
        dynamic_cooling (bool, optional): Whether to use dynamic cooling. Defaults to True.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    Attributes:
        init_temperature (float): The initial temperature.
        stopping_temperature (float): The stopping temperature.
        cooling_rate (float): The cooling rate.
        dynamic_cooling (bool): Whether to use dynamic cooling.

    Example:
        >>> from opt.classical.simulated_annealing import SimulatedAnnealing
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SimulatedAnnealing(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SimulatedAnnealing(
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
        init_temperature: float = 1000,
        stopping_temperature: float = 1e-8,
        cooling_rate: float = 0.99,
        *,
        dynamic_cooling: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize the SimulatedAnnealing class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.init_temperature = init_temperature
        self.stopping_temperature = stopping_temperature
        self.cooling_rate = cooling_rate
        self.dynamic_cooling = dynamic_cooling

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the simulated annealing optimization.

        Returns:
            tuple[np.ndarray, float]: The best solution found and its corresponding cost.

        """
        best_solution: np.ndarray = np.empty(self.dim)
        best_cost = np.inf

        for _ in range(self.population_size):
            current_solution = np.random.default_rng(self.seed).uniform(
                self.lower_bound, self.upper_bound, self.dim
            )
            current_cost = self.func(current_solution)

            if best_solution is None or current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

            temperature = self.init_temperature

            for _ in range(self.max_iter):
                new_solution = current_solution + np.random.default_rng(
                    self.seed
                ).uniform(-1, 1, self.dim)
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                new_cost = self.func(new_solution)

                delta_cost = new_cost - current_cost

                if delta_cost < 0 or np.random.default_rng(self.seed).random() < np.exp(
                    -delta_cost / temperature
                ):
                    self.seed += 1
                    current_solution = new_solution
                    current_cost = new_cost

                    if current_cost < best_cost:
                        best_solution = current_solution
                        best_cost = current_cost

                if self.dynamic_cooling:
                    temperature *= self.cooling_rate

                if temperature < self.stopping_temperature:
                    break

        return best_solution, best_cost


if __name__ == "__main__":
    # Create a SimulatedAnnealing object
    optimizer = SimulatedAnnealing(
        shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )

    # Perform the optimization

    best_solution, best_cost = optimizer.search()
    print(f"Best solution: {best_solution}")
    print(f"Best cost: {best_cost}")
