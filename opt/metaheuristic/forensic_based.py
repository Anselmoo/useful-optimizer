"""Forensic-Based Investigation Optimization.

Implementation based on:
Chou, J.S. & Nguyen, N.M. (2020).
FBI inspired meta-optimization.
Applied Soft Computing, 93, 106339.

The algorithm mimics the investigation process used by forensic
investigators, including evidence analysis and suspect tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class ForensicBasedInvestigationOptimizer(AbstractOptimizer):
    """Forensic-Based Investigation Optimization Algorithm.

    Simulates forensic investigation processes including:
    - Investigation phase: Gathering and analyzing evidence
    - Pursuit phase: Tracking and cornering suspects

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of investigators.
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
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Forensic-Based Investigation Optimization.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize investigator positions
        positions = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            (self.population_size, self.dim),
        )

        # Evaluate fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Best solution (prime suspect)
        best_idx = np.argmin(fitness)
        best_solution = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Mean position (investigation center)
        mean_position = np.mean(positions, axis=0)

        for iteration in range(self.max_iter):
            # Probability of investigation (decreases over time)
            p_investigation = 0.5 * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                r = np.random.rand()

                if r < p_investigation:
                    # Investigation phase (exploration)
                    # A - collecting evidence from crime scene

                    # Randomly select other investigators for teamwork
                    r1, r2 = np.random.choice(
                        self.population_size, size=2, replace=False
                    )
                    while r1 == i or r2 == i:
                        r1, r2 = np.random.choice(
                            self.population_size, size=2, replace=False
                        )

                    # Evidence analysis with random factor
                    beta = np.random.rand()
                    new_position = (
                        positions[i]
                        + beta * (positions[r1] - positions[r2])
                        + (1 - beta) * np.random.randn(self.dim)
                        * (mean_position - positions[i])
                    )
                else:
                    # Pursuit phase (exploitation)
                    # B - tracking the suspect

                    # Probability factor for pursuit
                    r_pursuit = np.random.rand()

                    if r_pursuit < 0.5:
                        # Direct pursuit toward best solution
                        alpha = 2 * np.random.rand() - 1
                        new_position = (
                            best_solution
                            + alpha * (best_solution - positions[i])
                        )
                    else:
                        # Coordinated team pursuit
                        team_idx = np.random.randint(self.population_size)
                        teammate = positions[team_idx]

                        gamma = np.random.rand()
                        new_position = (
                            positions[i]
                            + gamma * (best_solution - positions[i])
                            + (1 - gamma) * (teammate - positions[i])
                        )

                # Boundary handling
                new_position = np.clip(
                    new_position, self.lower_bound, self.upper_bound
                )

                # Evaluate new position
                new_fitness = self.func(new_position)

                # Greedy selection
                if new_fitness < fitness[i]:
                    positions[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

            # Update mean position (investigation center)
            mean_position = np.mean(positions, axis=0)

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = ForensicBasedInvestigationOptimizer(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=2.768,
        dim=2,
        max_iter=100,
        population_size=30,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
