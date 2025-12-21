"""Marine Predators Algorithm (MPA).

This module implements the Marine Predators Algorithm, a nature-inspired
metaheuristic based on the foraging strategy of ocean predators.

The algorithm mimics the Lévy and Brownian motion strategies used by marine
predators when hunting prey, with the choice of movement depending on the
velocity ratio between predator and prey.

Reference:
    Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
    Marine Predators Algorithm: A nature-inspired metaheuristic.
    Expert Systems with Applications, 152, 113377.
    DOI: 10.1016/j.eswa.2020.113377

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = MarinePredatorsOptimizer(
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
    population_size (int): Number of prey in the population.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for Marine Predators Algorithm
_FADs_EFFECT_PROB = 0.2  # Fish Aggregating Devices effect probability
_FADs_CONSTRUCTION_THRESHOLD = 0.5  # Threshold for FADs construction vs destruction
_PHASE_TRANSITION_1 = 1 / 3  # First phase transition point
_PHASE_TRANSITION_2 = 2 / 3  # Second phase transition point
_LEVY_BETA = 1.5  # Lévy flight parameter


class MarinePredatorsOptimizer(AbstractOptimizer):
    """Marine Predators Algorithm.

    This optimizer mimics ocean predator-prey interaction:
    - Phase 1 (high velocity ratio): Prey moves faster - Brownian motion
    - Phase 2 (unit velocity ratio): Both predator and prey explore - mixed motion
    - Phase 3 (low velocity ratio): Predator moves faster - Lévy flight
    - FADs effect: Fish Aggregating Devices provide additional exploration

    Attributes:
        seed (int): Random seed for reproducibility.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        population_size (int): Number of prey.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum iterations.
        func (Callable): Objective function to minimize.
        fads (float): FADs effect probability.


    Example:
        >>> from opt.swarm_intelligence.marine_predators_algorithm import MarinePredatorsOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = MarinePredatorsOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = MarinePredatorsOptimizer(
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
        fads: float = _FADs_EFFECT_PROB,
    ) -> None:
        """Initialize the Marine Predators Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Problem dimensionality.
            max_iter: Maximum iterations.
            seed: Random seed.
            population_size: Number of prey.
            fads: Fish Aggregating Devices effect probability.
        """
        super().__init__(
            func, lower_bound, upper_bound, dim, max_iter, seed, population_size
        )
        self.fads = fads

    def _levy_flight(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Generate Lévy flight steps.

        Args:
            rng: Random number generator.
            size: Size of the step vector.

        Returns:
            Lévy flight step vector.
        """
        sigma = (
            math.gamma(1 + _LEVY_BETA)
            * np.sin(np.pi * _LEVY_BETA / 2)
            / (
                math.gamma((1 + _LEVY_BETA) / 2)
                * _LEVY_BETA
                * 2 ** ((_LEVY_BETA - 1) / 2)
            )
        ) ** (1 / _LEVY_BETA)

        u = rng.normal(0, sigma, size)
        v = rng.normal(0, 1, size)

        return u / (np.abs(v) ** (1 / _LEVY_BETA))

    def _brownian_motion(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Generate Brownian motion steps.

        Args:
            rng: Random number generator.
            size: Size of the step vector.

        Returns:
            Brownian motion step vector.
        """
        return rng.normal(0, 1, size)

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Marine Predators Algorithm.

        Returns:
            Tuple containing:
                - best_solution: The best solution found (numpy array).
                - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize prey population
        prey = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(p) for p in prey])

        # Find top predator (elite)
        best_idx = np.argmin(fitness)
        elite = prey[best_idx].copy()
        elite_fitness = fitness[best_idx]

        # Create Elite matrix (all rows are copies of elite)
        elite_matrix = np.tile(elite, (self.population_size, 1))

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Calculate CF (control factor)
            cf = (1 - iteration / self.max_iter) ** (2 * iteration / self.max_iter)

            # Determine phase
            progress = iteration / self.max_iter

            # Update each prey position
            for i in range(self.population_size):
                r = rng.random(self.dim)
                step_size = np.zeros(self.dim)

                if progress < _PHASE_TRANSITION_1:
                    # Phase 1: High velocity ratio (prey moves faster) - Brownian
                    rb = self._brownian_motion(rng, self.dim)
                    step_size = rb * (elite_matrix[i] - rb * prey[i])
                    prey[i] = prey[i] + 0.5 * r * step_size

                elif progress < _PHASE_TRANSITION_2:
                    # Phase 2: Unit velocity ratio - mixed exploration
                    if i < self.population_size // 2:
                        # First half: Lévy based on prey
                        rl = self._levy_flight(rng, self.dim)
                        step_size = rl * (elite_matrix[i] - rl * prey[i])
                        prey[i] = prey[i] + 0.5 * r * step_size
                    else:
                        # Second half: Brownian based on elite
                        rb = self._brownian_motion(rng, self.dim)
                        step_size = rb * (rb * elite_matrix[i] - prey[i])
                        prey[i] = elite_matrix[i] + 0.5 * cf * step_size

                else:
                    # Phase 3: Low velocity ratio (predator faster) - Lévy
                    rl = self._levy_flight(rng, self.dim)
                    step_size = rl * (rl * elite_matrix[i] - prey[i])
                    prey[i] = elite_matrix[i] + 0.5 * cf * step_size

                # Ensure bounds
                prey[i] = np.clip(prey[i], self.lower_bound, self.upper_bound)

            # FADs effect (Fish Aggregating Devices)
            if rng.random() < self.fads:
                # Eddy formation and FADs effect
                r = rng.random()
                u = np.ones((self.population_size, self.dim)) * (
                    rng.random((self.population_size, self.dim)) < self.fads
                )

                if r < _FADs_CONSTRUCTION_THRESHOLD:
                    # FADs construction
                    indices = rng.permutation(self.population_size)
                    prey = (
                        prey
                        + cf
                        * (
                            self.lower_bound
                            + rng.random((self.population_size, self.dim))
                            * (self.upper_bound - self.lower_bound)
                        )
                        * u
                    )
                else:
                    # FADs destruction
                    indices = rng.permutation(self.population_size)
                    prey = prey + (self.fads * (1 - r) + r) * (
                        prey[
                            indices[: self.population_size // 2].repeat(2)[
                                : self.population_size
                            ]
                        ]
                        - prey[
                            indices[self.population_size // 2 :].repeat(2)[
                                : self.population_size
                            ]
                        ]
                    )

                # Ensure bounds after FADs effect
                prey = np.clip(prey, self.lower_bound, self.upper_bound)

            # Update fitness
            fitness = np.array([self.func(p) for p in prey])

            # Update elite
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < elite_fitness:
                elite = prey[best_idx].copy()
                elite_fitness = fitness[best_idx]
                elite_matrix = np.tile(elite, (self.population_size, 1))

        return elite, elite_fitness


if __name__ == "__main__":
    # Test with shifted Ackley function
    optimizer = MarinePredatorsOptimizer(
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
