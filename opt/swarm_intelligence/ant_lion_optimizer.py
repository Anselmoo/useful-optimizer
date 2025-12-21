"""Ant Lion Optimizer (ALO) Algorithm.

This module implements the Ant Lion Optimizer algorithm, a nature-inspired
metaheuristic based on the hunting mechanism of antlions.

Antlions dig cone-shaped pits in sand and wait for ants to fall in. When an ant
falls into the pit, the antlion throws sand outward to prevent escape. This hunting
mechanism is mathematically modeled for optimization.

Reference:
    Mirjalili, S. (2015). The Ant Lion Optimizer.
    Advances in Engineering Software, 83, 80-98.
    DOI: 10.1016/j.advengsoft.2015.01.010

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = AntLionOptimizer(
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
    population_size (int): Number of ants in the population.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


_RANDOM_WALK_THRESHOLD = 0.5


class AntLionOptimizer(AbstractOptimizer):
    """Ant Lion Optimizer Algorithm.

    This optimizer mimics the hunting behavior of antlions:
    - Ants randomly walk in the search space
    - Antlions build traps (pits) based on their fitness
    - Ants walk around antlions, influenced by trap radius
    - Elite antlion (best solution) has additional influence
    - Trap radius shrinks over iterations (intensification)

    Attributes:
        seed (int): Random seed for reproducibility.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        population_size (int): Number of ants.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum iterations.
        func (Callable): Objective function to minimize.
    """

    def _random_walk(
        self, rng: np.random.Generator, max_iter: int, dim: int
    ) -> np.ndarray:
        """Generate random walk sequence.

        Args:
            rng: Random number generator.
            max_iter: Number of walk steps.
            dim: Dimensionality.

        Returns:
            Cumulative random walk array of shape (max_iter, dim).
        """
        # Generate random steps: -1 or +1 based on random threshold
        steps = (
            2 * (rng.random((max_iter, dim)) > _RANDOM_WALK_THRESHOLD).astype(float) - 1
        )
        return np.cumsum(steps, axis=0)

    def _normalize_walk(
        self, walk: np.ndarray, lower: np.ndarray, upper: np.ndarray, iteration: int
    ) -> np.ndarray:
        """Normalize random walk to given bounds.

        Args:
            walk: Random walk array.
            lower: Lower bounds for normalization.
            upper: Upper bounds for normalization.
            iteration: Current iteration index.

        Returns:
            Normalized position at given iteration.
        """
        min_walk = walk.min(axis=0)
        max_walk = walk.max(axis=0)

        # Avoid division by zero
        range_walk = max_walk - min_walk
        range_walk = np.where(range_walk == 0, 1, range_walk)

        # Normalize to [lower, upper]
        normalized = (walk[iteration] - min_walk) / range_walk
        return lower + normalized * (upper - lower)

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Ant Lion Optimizer algorithm.

        Returns:
            Tuple containing:
                - best_solution: The best solution found (numpy array).
                - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize ant and antlion populations
        ants = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        antlions = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate fitness
        ant_fitness = np.array([self.func(ant) for ant in ants])
        antlion_fitness = np.array([self.func(al) for al in antlions])

        # Find elite antlion
        elite_idx = np.argmin(antlion_fitness)
        elite_antlion = antlions[elite_idx].copy()
        elite_fitness = antlion_fitness[elite_idx]

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Decrease trap boundary (intensification)
            # I ratio decreases from 1 to 10^-6 based on iteration
            w = 2 if iteration > 0.1 * self.max_iter else 1
            w = 3 if iteration > 0.5 * self.max_iter else w
            w = 4 if iteration > 0.75 * self.max_iter else w
            w = 5 if iteration > 0.9 * self.max_iter else w
            w = 6 if iteration > 0.95 * self.max_iter else w

            i_ratio = 10**w * (iteration / self.max_iter)

            for i in range(self.population_size):
                # Select antlion using roulette wheel selection
                # Convert to selection probabilities (lower fitness = higher prob)
                inv_fitness = 1 / (1 + antlion_fitness - antlion_fitness.min())
                probs = inv_fitness / inv_fitness.sum()
                selected_idx = rng.choice(self.population_size, p=probs)
                selected_antlion = antlions[selected_idx]

                # Calculate trap boundaries (shrink over iterations)
                c = self.lower_bound / i_ratio if i_ratio > 0 else self.lower_bound
                d = self.upper_bound / i_ratio if i_ratio > 0 else self.upper_bound

                # Bounds around selected antlion
                lb_antlion = selected_antlion + c
                ub_antlion = selected_antlion + d

                # Bounds around elite antlion
                lb_elite = elite_antlion + c
                ub_elite = elite_antlion + d

                # Clip bounds to search space
                lb_antlion = np.clip(lb_antlion, self.lower_bound, self.upper_bound)
                ub_antlion = np.clip(ub_antlion, self.lower_bound, self.upper_bound)
                lb_elite = np.clip(lb_elite, self.lower_bound, self.upper_bound)
                ub_elite = np.clip(ub_elite, self.lower_bound, self.upper_bound)

                # Random walks around antlion and elite
                walk_antlion = self._random_walk(rng, self.max_iter, self.dim)
                walk_elite = self._random_walk(rng, self.max_iter, self.dim)

                # Normalize walks
                ra = self._normalize_walk(
                    walk_antlion, lb_antlion, ub_antlion, iteration
                )
                re = self._normalize_walk(walk_elite, lb_elite, ub_elite, iteration)

                # Update ant position (average of walks)
                ants[i] = (ra + re) / 2

                # Ensure bounds
                ants[i] = np.clip(ants[i], self.lower_bound, self.upper_bound)

                # Update ant fitness
                ant_fitness[i] = self.func(ants[i])

            # Update antlions: replace if ant is better
            for i in range(self.population_size):
                if ant_fitness[i] < antlion_fitness[i]:
                    antlions[i] = ants[i].copy()
                    antlion_fitness[i] = ant_fitness[i]

            # Update elite antlion
            current_best_idx = np.argmin(antlion_fitness)
            if antlion_fitness[current_best_idx] < elite_fitness:
                elite_antlion = antlions[current_best_idx].copy()
                elite_fitness = antlion_fitness[current_best_idx]

        return elite_antlion, elite_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(AntLionOptimizer)
