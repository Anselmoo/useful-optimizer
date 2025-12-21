"""Harris Hawks Optimization (HHO) Algorithm.

This module implements the Harris Hawks Optimization algorithm, a population-based
metaheuristic inspired by the cooperative hunting behavior of Harris hawks in nature.

The algorithm simulates the surprise pounce (or seven kills) strategy where
hawks cooperate to catch prey. It includes exploration and exploitation phases
with different attacking strategies based on the escaping energy of prey.

Reference:
    Heidari, A.A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M., & Chen, H.
    (2019). Harris hawks optimization: Algorithm and applications.
    Future Generation Computer Systems, 97, 849-872.
    DOI: 10.1016/j.future.2019.02.028

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = HarrisHawksOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=10,
    ...     population_size=30,
    ...     max_iter=500,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
    >>> print(f"Best fitness: {best_fitness}")

Attributes:
    func (Callable): The objective function to minimize.
    lower_bound (float): Lower bound of the search space.
    upper_bound (float): Upper bound of the search space.
    dim (int): Dimensionality of the search space.
    population_size (int): Number of hawks in the population.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

import math

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


# Algorithm-specific constants (from original paper)
_EXPLORATION_THRESHOLD = 1.0  # |E| >= 1 triggers exploration
_SOFT_BESIEGE_THRESHOLD = 0.5  # |E| >= 0.5 triggers soft besiege
_RANDOM_THRESHOLD = 0.5  # Threshold for random decisions


class HarrisHawksOptimizer(AbstractOptimizer):
    """Harris Hawks Optimization Algorithm.

    This optimizer mimics the hunting behavior of Harris hawks, including:
    - Exploration phase: Hawks perch randomly based on other family members
    - Exploitation phase: Surprise pounce with soft/hard besiege strategies
    - Rapid dives: Levy flight-based movements for escaping prey

    The transition between exploration and exploitation is controlled by
    the escaping energy E, which decreases over iterations.

    Attributes:
        seed (int): Random seed for reproducibility.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        population_size (int): Number of hawks.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum iterations.
        func (Callable): Objective function to minimize.
    """

    def _levy_flight(self, rng: np.random.Generator, dim: int) -> np.ndarray:
        """Generate Levy flight step using Mantegna's algorithm.

        Args:
            rng: NumPy random generator.
            dim: Dimensionality of the step.

        Returns:
            Levy flight step vector.
        """
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = rng.normal(0, sigma, dim)
        v = rng.normal(0, 1, dim)
        return u / (np.abs(v) ** (1 / beta))

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Harris Hawks Optimization algorithm.

        Returns:
            Tuple containing:
                - best_solution: The best solution found (numpy array).
                - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize hawk population
        hawks = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(hawk) for hawk in hawks])

        # Find initial prey (best solution)
        best_idx = np.argmin(fitness)
        prey = hawks[best_idx].copy()
        prey_fitness = fitness[best_idx]

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Update escaping energy E (decreases from 2 to 0)
            e0 = 2 * rng.random() - 1  # Initial energy in [-1, 1]
            escaping_energy = 2 * e0 * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                q = rng.random()
                r = rng.random()

                if abs(escaping_energy) >= _EXPLORATION_THRESHOLD:
                    # Exploration phase
                    if q >= _RANDOM_THRESHOLD:
                        # Perch based on random tall tree (random hawk)
                        rand_idx = rng.integers(0, self.population_size)
                        hawks[i] = hawks[rand_idx] - rng.random() * abs(
                            hawks[rand_idx] - 2 * rng.random() * hawks[i]
                        )
                    else:
                        # Perch on random tall tree on the edge of home territory
                        hawks[i] = (prey - hawks.mean(axis=0)) - rng.random() * (
                            self.lower_bound
                            + rng.random() * (self.upper_bound - self.lower_bound)
                        )
                # Exploitation phase - different strategies based on |E| and r
                elif r >= _RANDOM_THRESHOLD:
                    # Soft besiege (prey has energy to escape)
                    if abs(escaping_energy) >= _SOFT_BESIEGE_THRESHOLD:
                        # Soft besiege
                        jump_strength = 2 * (1 - rng.random())
                        hawks[i] = prey - escaping_energy * abs(
                            jump_strength * prey - hawks[i]
                        )
                    else:
                        # Hard besiege
                        jump_strength = 2 * (1 - rng.random())
                        hawks[i] = prey - escaping_energy * abs(prey - hawks[i])
                # Progressive rapid dives with Levy flight
                elif abs(escaping_energy) >= _SOFT_BESIEGE_THRESHOLD:
                    # Soft besiege with progressive rapid dives
                    jump_strength = 2 * (1 - rng.random())
                    y = prey - escaping_energy * abs(jump_strength * prey - hawks[i])
                    y = np.clip(y, self.lower_bound, self.upper_bound)

                    if self.func(y) < fitness[i]:
                        hawks[i] = y
                    else:
                        # Levy flight
                        z = y + rng.random(self.dim) * self._levy_flight(rng, self.dim)
                        z = np.clip(z, self.lower_bound, self.upper_bound)
                        if self.func(z) < fitness[i]:
                            hawks[i] = z
                else:
                    # Hard besiege with progressive rapid dives
                    jump_strength = 2 * (1 - rng.random())
                    y = prey - escaping_energy * abs(
                        jump_strength * prey - hawks.mean(axis=0)
                    )
                    y = np.clip(y, self.lower_bound, self.upper_bound)

                    if self.func(y) < fitness[i]:
                        hawks[i] = y
                    else:
                        # Levy flight
                        z = y + rng.random(self.dim) * self._levy_flight(rng, self.dim)
                        z = np.clip(z, self.lower_bound, self.upper_bound)
                        if self.func(z) < fitness[i]:
                            hawks[i] = z

                # Ensure bounds
                hawks[i] = np.clip(hawks[i], self.lower_bound, self.upper_bound)

                # Update fitness
                fitness[i] = self.func(hawks[i])

                # Update prey (best solution)
                if fitness[i] < prey_fitness:
                    prey = hawks[i].copy()
                    prey_fitness = fitness[i]

        return prey, prey_fitness


if __name__ == "__main__":
    # Test with shifted Ackley function
    optimizer = HarrisHawksOptimizer(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=2.768,
        dim=2,
        population_size=30,
        max_iter=500,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
