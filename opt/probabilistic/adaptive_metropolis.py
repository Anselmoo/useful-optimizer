"""Simulated Annealing with Adaptive Metropolis.

This module implements Simulated Annealing enhanced with Adaptive Metropolis
proposal distribution, a probabilistic optimization method.

The algorithm adapts the proposal covariance based on the history of
accepted samples, improving exploration efficiency.

Reference:
    Haario, H., Saksman, E., & Tamminen, J. (2001).
    An adaptive Metropolis algorithm.
    Bernoulli, 7(2), 223-242.
    DOI: 10.2307/3318737

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = AdaptiveMetropolisOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-2.768,
    ...     upper_bound=2.768,
    ...     dim=2,
    ...     max_iter=1000,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class AdaptiveMetropolisOptimizer(AbstractOptimizer):
    """Adaptive Metropolis optimization algorithm.

    This algorithm uses:
    1. Metropolis-Hastings sampling with Gaussian proposals
    2. Adaptive covariance estimation from sample history
    3. Temperature annealing for optimization

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        initial_temp: Starting temperature.
        final_temp: Final temperature.
        adaptation_start: Iteration to start adaptation.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        initial_temp: float = 10.0,
        final_temp: float = 0.01,
        adaptation_start: int = 100,
    ) -> None:
        """Initialize Adaptive Metropolis Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum iterations. Defaults to 1000.
            initial_temp: Starting temperature. Defaults to 10.0.
            final_temp: Final temperature. Defaults to 0.01.
            adaptation_start: When to start adaptation. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.adaptation_start = adaptation_start

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Adaptive Metropolis optimization.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize
        current = np.random.uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        current_fitness = self.func(current)

        best_solution = current.copy()
        best_fitness = current_fitness

        # Initial covariance (diagonal)
        scale = (self.upper_bound - self.lower_bound) / 10
        cov = scale**2 * np.eye(self.dim)

        # Scaling factor for adaptive covariance
        s_d = 2.4**2 / self.dim  # Optimal scaling
        epsilon = 1e-6  # Small regularization

        # Sample history for covariance estimation
        sample_history = [current.copy()]
        sample_mean = current.copy()

        for iteration in range(self.max_iter):
            # Compute temperature
            t = iteration / self.max_iter
            temperature = self.initial_temp * (
                self.final_temp / self.initial_temp
            ) ** t

            # Generate proposal
            if iteration < self.adaptation_start:
                # Use initial covariance
                proposal = np.random.multivariate_normal(current, cov)
            else:
                # Use adapted covariance with small regularization
                adapted_cov = s_d * cov + s_d * epsilon * np.eye(self.dim)
                proposal = np.random.multivariate_normal(current, adapted_cov)

            # Boundary handling (reflection)
            proposal = np.clip(proposal, self.lower_bound, self.upper_bound)
            proposal_fitness = self.func(proposal)

            # Metropolis acceptance criterion
            delta = (proposal_fitness - current_fitness) / temperature
            if delta < 0 or np.random.random() < np.exp(-delta):
                current = proposal
                current_fitness = proposal_fitness

                # Update best
                if current_fitness < best_fitness:
                    best_solution = current.copy()
                    best_fitness = current_fitness

            # Update sample history and covariance
            sample_history.append(current.copy())
            n = len(sample_history)

            # Update running mean
            old_mean = sample_mean.copy()
            sample_mean = old_mean + (current - old_mean) / n

            # Update covariance (Welford's online algorithm)
            if n >= 2:
                cov = (
                    (n - 2) / (n - 1) * cov
                    + np.outer(old_mean, old_mean)
                    - n / (n - 1) * np.outer(sample_mean, sample_mean)
                    + 1 / (n - 1) * np.outer(current, current)
                )

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = AdaptiveMetropolisOptimizer(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=2.768,
        dim=2,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
