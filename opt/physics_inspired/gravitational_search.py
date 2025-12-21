"""Gravitational Search Algorithm (GSA).

This module implements the Gravitational Search Algorithm, a physics-inspired
metaheuristic based on Newton's law of gravity and laws of motion.

Objects (solutions) attract each other with gravitational forces proportional
to their mass (fitness) and inversely proportional to distance. Heavier masses
(better solutions) attract lighter masses (worse solutions).

Reference:
    Rashedi, E., Nezamabadi-Pour, H., & Saryazdi, S. (2009). GSA: A Gravitational
    Search Algorithm. Information Sciences, 179(13), 2232-2248.
    DOI: 10.1016/j.ins.2009.03.004

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = GravitationalSearchOptimizer(
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
    population_size (int): Number of agents in the population.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for gravitational search
_GRAVITATIONAL_CONSTANT_INITIAL = 100.0  # G0
_GRAVITATIONAL_DECAY_RATE = 20.0  # Alpha for G decay
_EPSILON = 1e-16  # Small value to avoid division by zero
_KBEST_DECAY_EXPONENT = 1.0  # Controls decrease rate of Kbest


class GravitationalSearchOptimizer(AbstractOptimizer):
    """Gravitational Search Algorithm.

    This optimizer simulates gravitational interaction between masses:
    - Each agent (solution) has mass proportional to its fitness
    - Better solutions have higher mass
    - Agents attract each other based on Newton's law of gravity
    - Gravitational constant G decreases over time for convergence
    - Only K-best agents exert forces (K decreases over time)

    Attributes:
        seed (int): Random seed for reproducibility.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        population_size (int): Number of agents.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum iterations.
        func (Callable): Objective function to minimize.
        g0 (float): Initial gravitational constant.
        alpha (float): Decay rate for gravitational constant.
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
        g0: float = _GRAVITATIONAL_CONSTANT_INITIAL,
        alpha: float = _GRAVITATIONAL_DECAY_RATE,
    ) -> None:
        """Initialize the Gravitational Search Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Problem dimensionality.
            max_iter: Maximum iterations.
            seed: Random seed.
            population_size: Number of agents.
            g0: Initial gravitational constant.
            alpha: Decay rate for gravitational constant.
        """
        super().__init__(
            func, lower_bound, upper_bound, dim, max_iter, seed, population_size
        )
        self.g0 = g0
        self.alpha = alpha

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Gravitational Search Algorithm.

        Returns:
            Tuple containing:
                - best_solution: The best solution found (numpy array).
                - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize agent population and velocities
        agents = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        velocities = np.zeros((self.population_size, self.dim))

        # Evaluate initial fitness
        fitness = np.array([self.func(agent) for agent in agents])

        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = agents[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Update gravitational constant (decreases over time)
            g = self.g0 * np.exp(-self.alpha * iteration / self.max_iter)

            # Calculate mass for each agent
            worst_fit = np.max(fitness)
            best_fit = np.min(fitness)
            fit_range = worst_fit - best_fit + _EPSILON

            # Mass proportional to fitness (minimization)
            m = (fitness - worst_fit) / fit_range
            # Invert for minimization (lower fitness = higher mass)
            m = np.exp(-m)
            # Normalize masses
            m = m / (np.sum(m) + _EPSILON)

            # Determine Kbest (number of agents that exert force)
            kbest = int(
                self.population_size
                - (self.population_size - 1)
                * (iteration / self.max_iter) ** _KBEST_DECAY_EXPONENT
            )
            kbest = max(1, kbest)

            # Sort agents by fitness and get K best indices
            sorted_indices = np.argsort(fitness)[:kbest]

            # Calculate forces on each agent
            forces = np.zeros((self.population_size, self.dim))

            for i in range(self.population_size):
                for j in sorted_indices:
                    if i != j:
                        # Distance between agents
                        distance_vec = agents[j] - agents[i]
                        distance = np.linalg.norm(distance_vec) + _EPSILON

                        # Gravitational force (random component for stochasticity)
                        force_magnitude = g * m[i] * m[j] / distance
                        forces[i] += rng.random() * force_magnitude * distance_vec

            # Calculate acceleration (F = ma, a = F/m)
            acceleration = forces / (m[:, np.newaxis] + _EPSILON)

            # Update velocities (with random component)
            velocities = (
                rng.random((self.population_size, self.dim)) * velocities + acceleration
            )

            # Update positions
            agents = agents + velocities

            # Ensure bounds
            agents = np.clip(agents, self.lower_bound, self.upper_bound)

            # Update fitness
            fitness = np.array([self.func(agent) for agent in agents])

            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = agents[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(GravitationalSearchOptimizer)
