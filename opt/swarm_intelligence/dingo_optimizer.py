"""Dingo Optimizer.

Implementation based on:
Peraza-Vázquez, H., Peña-Delgado, A.F., Echavarría-Castillo, G.,
Morales-Cepeda, A.B., Velasco-Álvarez, J. & Ruiz-Perez, F. (2021).
A Bio-Inspired Method for Engineering Design Optimization Inspired
by Dingoes Hunting Strategies.
Mathematical Problems in Engineering, 2021, 9107547.

The algorithm mimics the hunting strategies of dingoes, including
pack hunting, persecution, and attacking behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm parameters
_SURVIVAL_RATE = 0.3  # Survival rate of dingoes
_ATTACK_PROB = 0.5  # Probability of attack behavior


class DingoOptimizer(AbstractOptimizer):
    """Dingo Optimizer.

    Simulates the hunting strategies of dingoes, including:
    - Persecution: Chasing and cornering prey
    - Attack: Final strike on prey
    - Scavenger behavior: Finding alternative food sources

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of dingoes.
        survival_rate: Rate of survivors in each generation. Default 0.3.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        population_size: int = 30,
        survival_rate: float = _SURVIVAL_RATE,
    ) -> None:
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.survival_rate = survival_rate

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Dingo Optimizer.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize dingo pack
        positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Best solution (prey location)
        best_idx = np.argmin(fitness)
        best_solution = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Adaptive parameters
            a = 2 * (1 - iteration / self.max_iter)  # Decreasing from 2 to 0

            for i in range(self.population_size):
                r = np.random.rand()

                if r < _ATTACK_PROB:
                    # Persecution and attack behavior
                    r1, r2 = np.random.rand(2)
                    A = 2 * a * r1 - a  # Coefficient vector
                    C = 2 * r2  # Coefficient vector

                    # Distance to prey (best solution)
                    D = np.abs(C * best_solution - positions[i])

                    # Update position
                    new_position = best_solution - A * D
                else:
                    # Scavenger behavior - search for other food
                    # Group attack strategy
                    n_hunters = 3
                    hunters_idx = np.random.choice(
                        self.population_size,
                        size=min(n_hunters, self.population_size),
                        replace=False,
                    )
                    hunters = positions[hunters_idx]

                    # Move toward center of hunters
                    center = np.mean(hunters, axis=0)
                    r3 = np.random.rand()
                    new_position = center + r3 * (
                        self.upper_bound - self.lower_bound
                    ) * (2 * np.random.rand(self.dim) - 1) * (
                        1 - iteration / self.max_iter
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = self.func(new_position)

                # Greedy selection
                if new_fitness < fitness[i]:
                    positions[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

            # Survival selection - replace worst solutions
            n_survivors = int(self.population_size * self.survival_rate)
            sorted_idx = np.argsort(fitness)
            worst_idx = sorted_idx[-n_survivors:]

            # Generate new dingoes to replace worst
            for idx in worst_idx:
                if np.random.rand() < 0.5:
                    # Random initialization
                    positions[idx] = np.random.uniform(
                        self.lower_bound, self.upper_bound, self.dim
                    )
                else:
                    # Initialize near best
                    positions[idx] = best_solution + 0.1 * (
                        self.upper_bound - self.lower_bound
                    ) * np.random.randn(self.dim)
                    positions[idx] = np.clip(
                        positions[idx], self.lower_bound, self.upper_bound
                    )

                fitness[idx] = self.func(positions[idx])

                if fitness[idx] < best_fitness:
                    best_solution = positions[idx].copy()
                    best_fitness = fitness[idx]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(DingoOptimizer)
