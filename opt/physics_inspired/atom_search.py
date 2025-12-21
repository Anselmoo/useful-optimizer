"""Atom Search Optimization (ASO).

This module implements Atom Search Optimization, a physics-inspired
metaheuristic algorithm based on molecular dynamics simulation.

Reference:
    Zhao, W., Wang, L., & Zhang, Z. (2019).
    Atom search optimization and its application to solve a
    hydrogeologic parameter estimation problem.
    Knowledge-Based Systems, 163, 283-304.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for ASO algorithm
_ALPHA = 50  # Depth of Lennard-Jones potential
_BETA = 0.2  # Multiplier for attraction/repulsion
_G0 = 1.0  # Initial constraint factor
_EPSILON = 1e-10  # Small value to avoid division by zero


class AtomSearchOptimizer(AbstractOptimizer):
    """Atom Search Optimization implementation.

    ASO simulates the atomic motion based on molecular dynamics:
    - Atoms interact through Lennard-Jones potential
    - Interaction force depends on distance and mass (fitness)
    - Better atoms (lower fitness) have higher mass

    Attributes:
        func: The objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        population_size: Number of atoms.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.physics_inspired.atom_search import AtomSearchOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = AtomSearchOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = AtomSearchOptimizer(
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
        population_size: int = 50,
        max_iter: int = 500,
    ) -> None:
        """Initialize the ASO optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound for all dimensions.
            upper_bound: Upper bound for all dimensions.
            dim: Number of dimensions.
            population_size: Number of atoms.
            max_iter: Maximum iterations.
        """
        super().__init__(func, lower_bound, upper_bound, dim)
        self.population_size = population_size
        self.max_iter = max_iter

    def _calculate_mass(self, fitness: np.ndarray) -> np.ndarray:
        """Calculate mass of atoms based on fitness.

        Args:
            fitness: Fitness values of all atoms.

        Returns:
            Normalized mass values.
        """
        worst = np.max(fitness)
        best = np.min(fitness)

        if worst == best:
            return np.ones(len(fitness)) / len(fitness)

        # Mass is inversely related to fitness (better = higher mass)
        m = np.exp(-(fitness - best) / (worst - best + _EPSILON))
        return m / np.sum(m)

    def _calculate_constraint_factor(self, iteration: int) -> float:
        """Calculate constraint factor for force calculation.

        Args:
            iteration: Current iteration.

        Returns:
            Constraint factor value.
        """
        return np.exp(-20 * iteration / self.max_iter)

    def _lennard_jones_force(
        self, distance: float, depth: float, sigma: float
    ) -> float:
        """Calculate Lennard-Jones potential force.

        Args:
            distance: Distance between atoms.
            depth: Depth of potential well.
            sigma: Distance at which potential is zero.

        Returns:
            Force value (negative = attraction, positive = repulsion).
        """
        distance = max(distance, _EPSILON)

        ratio = sigma / distance
        ratio_6 = ratio**6
        ratio_12 = ratio_6**2

        return depth * (ratio_12 - ratio_6)

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the optimization algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population (atoms)
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Initialize velocities
        velocities = np.zeros((self.population_size, self.dim))

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Initialize best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Calculate search space diagonal for sigma
        diagonal = np.sqrt(self.dim * (self.upper_bound - self.lower_bound) ** 2)
        sigma = _BETA * diagonal

        for iteration in range(self.max_iter):
            # Calculate mass of atoms
            mass = self._calculate_mass(fitness)

            # Calculate constraint factor
            g = _G0 * self._calculate_constraint_factor(iteration)

            # Calculate interaction forces
            forces = np.zeros((self.population_size, self.dim))

            for i in range(self.population_size):
                for j in range(self.population_size):
                    if i == j:
                        continue

                    # Calculate distance
                    diff = population[j] - population[i]
                    distance = np.linalg.norm(diff)

                    if distance < _EPSILON:
                        continue

                    # Direction vector
                    direction = diff / distance

                    # Calculate force magnitude
                    force_mag = self._lennard_jones_force(distance, _ALPHA, sigma)

                    # Apply mass weighting
                    force = g * force_mag * mass[j] * direction

                    forces[i] += force

            # Update velocities and positions
            for i in range(self.population_size):
                # Update velocity
                rand = np.random.rand(self.dim)
                velocities[i] = rand * velocities[i] + forces[i]

                # Update position
                new_position = population[i] + velocities[i]

                # Boundary handling with reflection
                for d in range(self.dim):
                    if new_position[d] < self.lower_bound:
                        new_position[d] = self.lower_bound
                        velocities[i, d] *= -1
                    elif new_position[d] > self.upper_bound:
                        new_position[d] = self.upper_bound
                        velocities[i, d] *= -1

                # Evaluate and update
                new_fitness = self.func(new_position)
                population[i] = new_position
                fitness[i] = new_fitness

                # Update best if necessary
                if new_fitness < best_fitness:
                    best_solution = new_position.copy()
                    best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = AtomSearchOptimizer(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=2.768,
        dim=2,
        population_size=30,
        max_iter=100,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
