"""Slime Mould Algorithm (SMA) implementation.

This module implements the Slime Mould Algorithm, a nature-inspired
optimization algorithm based on the oscillation mode of slime mould
in nature during foraging.

Reference:
    Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020).
    Slime mould algorithm: A new method for stochastic optimization.
    Future Generation Computer Systems, 111, 300-323.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_Z = 0.03  # Random exploration threshold


class SlimeMouldAlgorithm(AbstractOptimizer):
    """Slime Mould Algorithm optimizer.

    The SMA simulates the oscillation behavior of slime mould:
    - Approaching food sources (exploitation)
    - Wrapping food using bio-oscillator (exploration)
    - Grabbing food based on concentration gradients

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of slime moulds (solutions).
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
        """Initialize the Slime Mould Algorithm.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of slime moulds (solutions).
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Slime Mould Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Find best and worst
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Main loop
        for iteration in range(self.max_iter):
            # Sort by fitness
            sorted_indices = np.argsort(fitness)
            worst_fitness = fitness[sorted_indices[-1]]

            # Update a parameter (decreases from 1 to 0, avoid arctanh(1))
            t_ratio = max(1e-10, 1 - (iteration + 1) / self.max_iter)
            a = np.arctanh(t_ratio)

            # Update b parameter (oscillates)
            b = 1 - iteration / self.max_iter

            for i in range(self.population_size):
                # Calculate weight
                if i < self.population_size // 2:
                    w = 1 + np.random.rand() * np.log10(
                        (best_fitness - fitness[sorted_indices[i]])
                        / (best_fitness - worst_fitness + 1e-10)
                        + 1
                    )
                else:
                    w = 1 - np.random.rand() * np.log10(
                        (best_fitness - fitness[sorted_indices[i]])
                        / (best_fitness - worst_fitness + 1e-10)
                        + 1
                    )

                # Update position
                p = np.tanh(np.abs(fitness[i] - best_fitness))
                vb = 2 * a * np.random.rand() - a  # [-a, a]
                vc = 2 * b * np.random.rand() - b  # [-b, b]

                r = np.random.rand()

                if r < _Z:
                    # Random exploration
                    new_position = np.random.uniform(
                        self.lower_bound, self.upper_bound, self.dim
                    )
                elif r < p:
                    # Food approaching behavior
                    rand_idx_a = np.random.randint(self.population_size)
                    rand_idx_b = np.random.randint(self.population_size)
                    new_position = best_solution + vb * (
                        w * population[rand_idx_a] - population[rand_idx_b]
                    )
                else:
                    # Wrapping behavior
                    new_position = vc * population[i]

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update
                new_fitness = self.func(new_position)

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(SlimeMouldAlgorithm)
