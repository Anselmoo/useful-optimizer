"""Harmony Search (HS) algorithm.

This module implements the Harmony Search optimization algorithm. Harmony Search is a
metaheuristic algorithm inspired by the improvisation process of musicians. It is
commonly used for solving optimization problems.

The HarmonySearch class is the main class that implements the algorithm. It takes an
objective function, lower and upper bounds of the search space, dimensionality of the
search space, and other optional parameters. The search method runs the optimization
process and returns the best solution found and its fitness value.

Example:
    optimizer = HarmonySearch(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=5000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    harmony_memory_accepting_rate (float): The rate at which the harmony memory is accepted.
    pitch_adjusting_rate (float): The rate at which the pitch is adjusted.
    bandwidth (float): The bandwidth for adjusting the pitch.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class HarmonySearch(AbstractOptimizer):
    """Harmony Search optimizer.

    This class implements the Harmony Search optimization algorithm.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The size of the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        harmony_memory_accepting_rate (float, optional): The rate at which the harmony memory is accepted. Defaults to 0.95.
        pitch_adjusting_rate (float, optional): The rate at which the pitch is adjusted. Defaults to 0.7.
        bandwidth (float, optional): The bandwidth for adjusting the pitch. Defaults to 0.01.
        seed (int | None, optional): The seed for the random number generator. Defaults to None.

    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 100,
        max_iter: int = 1000,
        harmony_memory_accepting_rate: float = 0.95,
        pitch_adjusting_rate: float = 0.7,
        bandwidth: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Initialize the HarmonySearch class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.harmony_memory_accepting_rate = harmony_memory_accepting_rate
        self.pitch_adjusting_rate = pitch_adjusting_rate
        self.bandwidth = bandwidth

    def _initialize(self) -> ndarray:
        """Initialize the harmony memory.

        Returns:
            ndarray: The initialized harmony memory.

        """
        return np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

    def _generate_new_solution(self, harmony_memory: ndarray) -> ndarray:
        """Generate a new solution based on the harmony memory.

        Args:
            harmony_memory (ndarray): The harmony memory.

        Returns:
            ndarray: The new solution.

        """
        new_solution = np.zeros(self.dim)
        for i in range(self.dim):
            self.seed += 1
            if (
                np.random.default_rng(self.seed).random()
                < self.harmony_memory_accepting_rate
            ):
                new_solution[i] = harmony_memory[
                    np.random.default_rng(self.seed).integers(self.population_size), i
                ]
                self.seed += 1
                if (
                    np.random.default_rng(self.seed).random()
                    < self.pitch_adjusting_rate
                ):
                    new_solution[i] += self.bandwidth * np.random.default_rng(
                        self.seed
                    ).uniform(-1, 1)
            else:
                new_solution[i] = np.random.default_rng(self.seed).uniform(
                    self.lower_bound, self.upper_bound
                )
        return new_solution

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Harmony Search optimization.

        Returns:
            tuple[np.ndarray, float]: The best solution found and its fitness value.

        """
        harmony_memory = self._initialize()
        fitness = np.apply_along_axis(self.func, 1, harmony_memory)
        best_idx = np.argmin(fitness)
        best_solution = harmony_memory[best_idx]
        best_fitness = fitness[best_idx]

        for _ in range(self.max_iter):
            new_solution = self._generate_new_solution(harmony_memory)
            new_fitness = self.func(new_solution)
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                harmony_memory[worst_idx] = new_solution
                fitness[worst_idx] = new_fitness
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(HarmonySearch)
