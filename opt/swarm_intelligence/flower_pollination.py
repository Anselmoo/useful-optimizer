"""Flower Pollination Algorithm (FPA) implementation.

This module implements the Flower Pollination Algorithm, a nature-inspired
metaheuristic optimization algorithm based on the pollination process of
flowering plants.

Reference:
    Yang, X.-S. (2012). Flower pollination algorithm for global optimization.
    In Unconventional Computation and Natural Computation (pp. 240-249).
    Springer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.special import gamma

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_BETA = 1.5  # Lévy flight exponent
_SWITCH_PROBABILITY = 0.8  # Probability of global pollination


class FlowerPollinationAlgorithm(AbstractOptimizer):
    """Flower Pollination Algorithm optimizer.

    The FPA mimics the pollination behavior of flowering plants where:
    - Global pollination (biotic) is carried by pollinators following Lévy flights
    - Local pollination (abiotic) occurs through self-pollination and wind

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of flowers (solutions).
        switch_probability: Probability of global pollination (default: 0.8).


    Example:
        >>> from opt.swarm_intelligence.flower_pollination import FlowerPollinationAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = FlowerPollinationAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = FlowerPollinationAlgorithm(
        ...     func=shifted_ackley,
        ...     dim=2,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     max_iter=10,
        ...     seed=42,
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
        max_iter: int,
        population_size: int = 25,
        switch_probability: float = _SWITCH_PROBABILITY,
    ) -> None:
        """Initialize the Flower Pollination Algorithm.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of flowers (solutions).
            switch_probability: Probability of global pollination.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.switch_probability = switch_probability

    def _levy_flight(self, dim: int) -> np.ndarray:
        """Generate Lévy flight step.

        Uses Mantegna's algorithm to approximate Lévy flights with
        a stability parameter of 1.5.

        Args:
            dim: Dimensionality for the step.

        Returns:
            Lévy flight step vector.
        """
        beta = _BETA

        # Calculate sigma using Mantegna's algorithm
        sigma_u = (
            gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        sigma_v = 1.0

        u = np.random.randn(dim) * sigma_u
        v = np.random.randn(dim) * sigma_v

        return u / (np.abs(v) ** (1 / beta))

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Flower Pollination Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population (flowers)
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Find best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Main loop
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                if np.random.rand() < self.switch_probability:
                    # Global pollination via Lévy flights
                    levy = self._levy_flight(self.dim)
                    new_position = population[i] + levy * (
                        best_solution - population[i]
                    )
                else:
                    # Local pollination
                    # Randomly select two different flowers
                    epsilon = np.random.rand()
                    j = np.random.randint(self.population_size)
                    k = np.random.randint(self.population_size)
                    while j == i:
                        j = np.random.randint(self.population_size)
                    while k == i or k == j:
                        k = np.random.randint(self.population_size)

                    new_position = population[i] + epsilon * (
                        population[j] - population[k]
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new solution
                new_fitness = self.func(new_position)

                # Greedy selection
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(FlowerPollinationAlgorithm)
