"""Sequential Monte Carlo Optimizer.

This module implements Sequential Monte Carlo (SMC) optimization,
a probabilistic method using importance sampling and particle resampling.

The algorithm maintains a population of weighted particles that
progressively focus on promising regions of the search space.

Reference:
    Del Moral, P., Doucet, A., & Jasra, A. (2006).
    Sequential Monte Carlo Samplers.
    Journal of the Royal Statistical Society: Series B, 68(3), 411-436.
    DOI: 10.1111/j.1467-9868.2006.00553.x

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = SequentialMonteCarloOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-2.768,
    ...     upper_bound=2.768,
    ...     dim=2,
    ...     population_size=50,
    ...     max_iter=100,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class SequentialMonteCarloOptimizer(AbstractOptimizer):
    """Sequential Monte Carlo optimization algorithm.

    This algorithm uses:
    1. Importance sampling with adaptive weighting
    2. Systematic resampling to focus on promising particles
    3. MCMC moves to maintain diversity

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of particles.
        max_iter: Maximum number of iterations.
        temperature_schedule: Temperature annealing schedule.


    Example:
        >>> from opt.probabilistic.sequential_monte_carlo import SequentialMonteCarloOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SequentialMonteCarloOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SequentialMonteCarloOptimizer(
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
        max_iter: int = 100,
        initial_temp: float = 10.0,
        final_temp: float = 0.1,
    ) -> None:
        """Initialize Sequential Monte Carlo Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of particles. Defaults to 50.
            max_iter: Maximum iterations. Defaults to 100.
            initial_temp: Starting temperature. Defaults to 10.0.
            final_temp: Final temperature. Defaults to 0.1.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.initial_temp = initial_temp
        self.final_temp = final_temp

    def _systematic_resample(self, weights: np.ndarray, n_samples: int) -> np.ndarray:
        """Perform systematic resampling.

        Args:
            weights: Normalized particle weights.
            n_samples: Number of samples to draw.

        Returns:
            Indices of resampled particles.
        """
        cumsum = np.cumsum(weights)
        u0 = np.random.random() / n_samples
        u = u0 + np.arange(n_samples) / n_samples
        indices = np.searchsorted(cumsum, u)
        return indices

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Sequential Monte Carlo optimization.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize particles uniformly
        particles = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(p) for p in particles])

        best_idx = np.argmin(fitness)
        best_solution = particles[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Initialize weights uniformly
        weights = np.ones(self.population_size) / self.population_size

        for iteration in range(self.max_iter):
            # Compute current temperature
            t = iteration / self.max_iter
            temperature = self.initial_temp * (self.final_temp / self.initial_temp) ** t

            # Compute importance weights based on fitness
            log_weights = -fitness / temperature
            log_weights -= np.max(log_weights)  # Numerical stability
            weights = np.exp(log_weights)
            weights /= np.sum(weights)

            # Effective sample size
            ess = 1.0 / np.sum(weights**2)

            # Resample if ESS is low
            if ess < self.population_size / 2:
                indices = self._systematic_resample(weights, self.population_size)
                particles = particles[indices]
                fitness = fitness[indices]
                weights = np.ones(self.population_size) / self.population_size

            # MCMC move step (Gaussian perturbation)
            scale = (self.upper_bound - self.lower_bound) * (1 - t) * 0.1

            for i in range(self.population_size):
                # Propose new particle
                proposal = particles[i] + np.random.normal(0, scale, self.dim)
                proposal = np.clip(proposal, self.lower_bound, self.upper_bound)
                proposal_fitness = self.func(proposal)

                # Metropolis acceptance
                delta = (proposal_fitness - fitness[i]) / temperature
                if delta < 0 or np.random.random() < np.exp(-delta):
                    particles[i] = proposal
                    fitness[i] = proposal_fitness

                    if proposal_fitness < best_fitness:
                        best_solution = proposal.copy()
                        best_fitness = proposal_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(SequentialMonteCarloOptimizer)
