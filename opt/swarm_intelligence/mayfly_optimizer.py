"""Mayfly Optimization Algorithm.

Implementation based on:
Zervoudakis, K. & Tsafarakis, S. (2020).
A mayfly optimization algorithm.
Computers & Industrial Engineering, 145, 106559.

The algorithm mimics the mating behavior of mayflies, including nuptial
dances performed by males to attract females and the swarm dynamics of
both male and female mayflies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_A1 = 1.0  # Cognitive coefficient for males
_A2 = 1.5  # Social coefficient for males
_A3 = 1.5  # Attraction coefficient for females
_BETA = 2.0  # Attraction exponent
_DANCE = 5.0  # Nuptial dance coefficient
_FL = 0.1  # Random flight coefficient
_G = 0.8  # Gravity coefficient


class MayflyOptimizer(AbstractOptimizer):
    """Mayfly Optimization Algorithm.

    Simulates the mating behavior of mayflies with gender-specific
    movement patterns. Males perform nuptial dances and are attracted
    to the best positions, while females move toward males.

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Total number of mayflies (half male, half female).
        a1: Male cognitive coefficient. Default 1.0.
        a2: Male social coefficient. Default 1.5.
        a3: Female attraction coefficient. Default 1.5.
        beta: Attraction exponent. Default 2.0.
        dance: Nuptial dance coefficient. Default 5.0.
        fl: Random flight coefficient. Default 0.1.
        g: Gravity coefficient. Default 0.8.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        population_size: int = 30,
        a1: float = _A1,
        a2: float = _A2,
        a3: float = _A3,
        beta: float = _BETA,
        dance: float = _DANCE,
        fl: float = _FL,
        g: float = _G,
    ) -> None:
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.n_males = population_size // 2
        self.n_females = population_size - self.n_males
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.beta = beta
        self.dance = dance
        self.fl = fl
        self.g = g

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Mayfly Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize male mayfly positions and velocities
        males = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.n_males, self.dim)
        )
        male_vel = np.zeros((self.n_males, self.dim))

        # Initialize female mayfly positions and velocities
        females = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.n_females, self.dim)
        )
        female_vel = np.zeros((self.n_females, self.dim))

        # Evaluate fitness
        male_fitness = np.array([self.func(m) for m in males])
        female_fitness = np.array([self.func(f) for f in females])

        # Personal bests for males
        male_pbest = males.copy()
        male_pbest_fitness = male_fitness.copy()

        # Global best
        all_positions = np.vstack([males, females])
        all_fitness = np.concatenate([male_fitness, female_fitness])
        best_idx = np.argmin(all_fitness)
        best_solution = all_positions[best_idx].copy()
        best_fitness = all_fitness[best_idx]

        for iteration in range(self.max_iter):
            # Update damping coefficient
            damp = 0.95 - 0.5 * (iteration / self.max_iter)

            # Update male positions
            for i in range(self.n_males):
                r1, r2 = np.random.rand(2)

                # Cognitive component
                cognitive = self.a1 * r1 * (male_pbest[i] - males[i])
                # Social component
                social = self.a2 * r2 * (best_solution - males[i])

                # Nuptial dance component
                if male_fitness[i] < best_fitness:
                    dance_component = self.dance * np.random.randn(self.dim)
                else:
                    dance_component = 0

                # Update velocity
                male_vel[i] = (
                    self.g * male_vel[i] + cognitive + social + dance_component
                )
                male_vel[i] *= damp

                # Update position
                males[i] = males[i] + male_vel[i]

                # Boundary handling
                males[i] = np.clip(males[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = self.func(males[i])
                male_fitness[i] = new_fitness

                # Update personal best
                if new_fitness < male_pbest_fitness[i]:
                    male_pbest[i] = males[i].copy()
                    male_pbest_fitness[i] = new_fitness

            # Update female positions
            for i in range(self.n_females):
                # Female moves toward corresponding male
                male_idx = i if i < self.n_males else i % self.n_males
                male_pos = males[male_idx]

                # Calculate distance
                distance = np.linalg.norm(females[i] - male_pos)

                if female_fitness[i] > male_fitness[male_idx]:
                    # Female is attracted to male
                    r3 = np.random.rand()
                    attraction = (
                        self.a3
                        * np.exp(-self.beta * distance**2)
                        * r3
                        * (male_pos - females[i])
                    )
                    female_vel[i] = self.g * female_vel[i] + attraction
                else:
                    # Random flight
                    female_vel[i] = self.g * female_vel[i] + self.fl * np.random.randn(
                        self.dim
                    )

                female_vel[i] *= damp

                # Update position
                females[i] = females[i] + female_vel[i]

                # Boundary handling
                females[i] = np.clip(females[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                female_fitness[i] = self.func(females[i])

            # Update global best
            for i in range(self.n_males):
                if male_fitness[i] < best_fitness:
                    best_solution = males[i].copy()
                    best_fitness = male_fitness[i]

            for i in range(self.n_females):
                if female_fitness[i] < best_fitness:
                    best_solution = females[i].copy()
                    best_fitness = female_fitness[i]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(MayflyOptimizer)
