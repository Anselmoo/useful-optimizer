"""Equilibrium Optimizer (EO).

This module implements the Equilibrium Optimizer, a physics-inspired metaheuristic
based on control volume mass balance models used to estimate dynamic and equilibrium
states.

The algorithm uses concepts from mass balance to describe concentration changes
in a control volume, simulating particles reaching equilibrium states.

Reference:
    Faramarzi, A., Heidarinejad, M., Stephens, B., & Mirjalili, S. (2020).
    Equilibrium optimizer: A novel optimization algorithm. Knowledge-Based Systems,
    191, 105190. DOI: 10.1016/j.knosys.2019.105190

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = EquilibriumOptimizer(
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
    population_size (int): Number of particles in the population.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for equilibrium optimizer
_A1 = 2.0  # Constant for generation rate control
_A2 = 1.0  # Constant for generation probability
_GP = 0.5  # Generation probability
_EQUILIBRIUM_POOL_SIZE = 4  # Number of best solutions in equilibrium pool
_MIN_POOL_IDX_2 = 1  # Minimum index for second equilibrium candidate
_MIN_POOL_IDX_3 = 2  # Minimum index for third equilibrium candidate
_MIN_POOL_IDX_4 = 3  # Minimum index for fourth equilibrium candidate


class EquilibriumOptimizer(AbstractOptimizer):
    """Equilibrium Optimizer.

    This optimizer mimics the mass balance in a control volume:
    - Particles represent candidate solutions
    - Equilibrium pool consists of best solutions found
    - Particles update using exponential decay toward equilibrium
    - Generation term provides exploration capability
    - Balance between exploration and exploitation controlled by time

    Attributes:
        seed (int): Random seed for reproducibility.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        population_size (int): Number of particles.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum iterations.
        func (Callable): Objective function to minimize.
        a1 (float): Generation rate control constant.
        a2 (float): Generation probability constant.
        gp (float): Generation probability.


    Example:
        >>> from opt.physics_inspired.equilibrium_optimizer import EquilibriumOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = EquilibriumOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = EquilibriumOptimizer(
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
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 100,
        a1: float = _A1,
        a2: float = _A2,
        gp: float = _GP,
    ) -> None:
        """Initialize the Equilibrium Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Problem dimensionality.
            max_iter: Maximum iterations.
            seed: Random seed.
            population_size: Number of particles.
            a1: Generation rate control constant.
            a2: Generation probability constant.
            gp: Generation probability.
        """
        super().__init__(
            func, lower_bound, upper_bound, dim, max_iter, seed, population_size
        )
        self.a1 = a1
        self.a2 = a2
        self.gp = gp

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Equilibrium Optimizer algorithm.

        Returns:
            Tuple containing:
                - best_solution: The best solution found (numpy array).
                - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize particle population
        particles = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(p) for p in particles])

        # Initialize equilibrium pool (4 best + average)
        sorted_indices = np.argsort(fitness)
        c_eq1 = particles[sorted_indices[0]].copy()
        c_eq2 = (
            particles[sorted_indices[1]].copy()
            if len(sorted_indices) > _MIN_POOL_IDX_2
            else c_eq1.copy()
        )
        c_eq3 = (
            particles[sorted_indices[2]].copy()
            if len(sorted_indices) > _MIN_POOL_IDX_3
            else c_eq1.copy()
        )
        c_eq4 = (
            particles[sorted_indices[3]].copy()
            if len(sorted_indices) > _MIN_POOL_IDX_4
            else c_eq1.copy()
        )
        c_eq_avg = (c_eq1 + c_eq2 + c_eq3 + c_eq4) / _EQUILIBRIUM_POOL_SIZE

        # Best solution tracking
        best_solution = c_eq1.copy()
        best_fitness = fitness[sorted_indices[0]]

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Time parameter (decreases from 1 to 0)
            t = (1 - iteration / self.max_iter) ** (self.a2 * iteration / self.max_iter)

            for i in range(self.population_size):
                # Randomly select equilibrium candidate
                eq_pool = [c_eq1, c_eq2, c_eq3, c_eq4, c_eq_avg]
                c_eq = eq_pool[rng.integers(0, 5)]

                # Generation rate control
                r = rng.random(self.dim)
                lambda_param = rng.random(self.dim)
                r1 = rng.random()
                r2 = rng.random()

                # Exponential term
                f = self.a1 * np.sign(r - 0.5) * (np.exp(-lambda_param * t) - 1)

                # Generation rate
                gcp = _GP * r1 if r2 >= self.gp else 0

                # Calculate G0 and G
                g0 = gcp * (c_eq - lambda_param * particles[i])
                g = g0 * f

                # Update particle position
                particles[i] = (
                    c_eq
                    + (particles[i] - c_eq) * f
                    + (g / (lambda_param * (self.upper_bound - self.lower_bound)))
                    * (1 - f)
                )

                # Ensure bounds
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                # Update fitness
                fitness[i] = self.func(particles[i])

            # Update equilibrium pool
            sorted_indices = np.argsort(fitness)

            # Update equilibrium candidates if better solutions found
            if fitness[sorted_indices[0]] < self.func(c_eq1):
                c_eq1 = particles[sorted_indices[0]].copy()
            if len(sorted_indices) > _MIN_POOL_IDX_2 and fitness[
                sorted_indices[1]
            ] < self.func(c_eq2):
                c_eq2 = particles[sorted_indices[1]].copy()
            if len(sorted_indices) > _MIN_POOL_IDX_3 and fitness[
                sorted_indices[2]
            ] < self.func(c_eq3):
                c_eq3 = particles[sorted_indices[2]].copy()
            if len(sorted_indices) > _MIN_POOL_IDX_4 and fitness[
                sorted_indices[3]
            ] < self.func(c_eq4):
                c_eq4 = particles[sorted_indices[3]].copy()

            c_eq_avg = (c_eq1 + c_eq2 + c_eq3 + c_eq4) / _EQUILIBRIUM_POOL_SIZE

            # Update best solution
            current_best_fitness = self.func(c_eq1)
            if current_best_fitness < best_fitness:
                best_solution = c_eq1.copy()
                best_fitness = current_best_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(EquilibriumOptimizer)
