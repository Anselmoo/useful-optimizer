"""Particle Filter Algorithm.

This module implements the Particle Filter algorithm. Particle filters, or Sequential
Monte Carlo (SMC) methods, are a set of on-line posterior density estimation algorithms
that estimate the posterior density of the state-space by directly implementing the
Bayesian recursion equations.

The main idea behind particle filters is to represent the posterior density function by
a set of random samples, or particles, and assign a weight to each particle that
represents the probability of that particle being sampled from the probability density
function.

Particle filters are particularly useful for non-linear and non-Gaussian estimation
problems.

Example:
    filter = ParticleFilter(func=state_transition_function, initial_state=[0, 0],
    num_particles=100)
    next_state = filter.predict()

Attributes:
    func (Callable): The state transition function.
    initial_state (List[float]): The initial state.
    num_particles (int): The number of particles.

Methods:
    predict(): Perform a prediction step in the particle filter algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ParticleFilter(AbstractOptimizer):
    """ParticleFilter is a class that implements the Particle Swarm Optimization algorithm for optimization problems.

    Parameters:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        population_size (int, optional): The number of particles in the population. Defaults to 100.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        inertia (float, optional): The inertia weight. Defaults to 0.7.
        cognitive (float, optional): The cognitive weight. Defaults to 1.5.
        social (float, optional): The social weight. Defaults to 1.5.
        seed (Optional[int], optional): The seed for the random number generator. Defaults to None.


    Example:
        >>> from opt.metaheuristic.particle_filter import ParticleFilter
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ParticleFilter(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ParticleFilter(
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
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 100,
        max_iter: int = 1000,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5,
        seed: int | None = None,
    ) -> None:
        """Initialize the ParticleFilter class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.max_iter = max_iter
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def search(self) -> tuple[np.ndarray, float]:
        """Run the Particle Swarm Optimization algorithm to find the optimal solution.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the global best position and
                the corresponding score.
        """
        # Initialize particles
        particles = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        velocities = np.zeros_like(particles)
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)

        # Initialize global best
        global_best_position: np.ndarray = np.array([])
        global_best_score = np.inf

        for _ in range(self.max_iter):
            self.seed += 1
            # Evaluate particles
            scores = np.apply_along_axis(self.func, 1, particles)

            # Update personal best
            mask = scores < personal_best_scores
            personal_best_positions[mask] = particles[mask]
            personal_best_scores[mask] = scores[mask]

            # Update global best
            min_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_idx] < global_best_score:
                global_best_score = personal_best_scores[min_idx]
                global_best_position = personal_best_positions[min_idx]

            # Update velocities and particles
            r1 = np.random.default_rng(self.seed + 1).random(
                (self.population_size, self.dim)
            )
            r2 = np.random.default_rng(self.seed + 2).random(
                (self.population_size, self.dim)
            )
            velocities = (
                self.inertia * velocities
                + self.cognitive * r1 * (personal_best_positions - particles)
                + self.social * r2 * (global_best_position - particles)
            )
            particles += velocities

            # Ensure particles are within bounds
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

        return global_best_position, global_best_score


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ParticleFilter)
