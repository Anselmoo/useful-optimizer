"""Soccer League Competition Algorithm.

This module implements the Soccer League Competition (SLC) algorithm,
a social-inspired metaheuristic based on soccer league dynamics.

The algorithm simulates soccer team behaviors including matches,
transfers, and training processes.

Reference:
    Moosavian, N., & Roodsari, B. K. (2014).
    Soccer League Competition Algorithm: A novel meta-heuristic algorithm for
    optimal design of water distribution networks.
    Swarm and Evolutionary Computation, 17, 14-24.
    DOI: 10.1016/j.swevo.2014.02.002

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = SoccerLeagueOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-2.768,
    ...     upper_bound=2.768,
    ...     dim=2,
    ...     population_size=30,
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


class SoccerLeagueOptimizer(AbstractOptimizer):
    """Soccer League Competition algorithm.

    This algorithm simulates soccer league behaviors:
    1. Match process - competition between teams
    2. Transfer window - player movement between teams
    3. Training - team improvement

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of teams.
        max_iter: Maximum number of seasons.
        num_teams: Number of teams per league.


    Example:
        >>> from opt.social_inspired.soccer_league_optimizer import SoccerLeagueOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SoccerLeagueOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SoccerLeagueOptimizer(
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
        population_size: int = 30,
        max_iter: int = 100,
        num_teams: int = 10,
    ) -> None:
        """Initialize Soccer League Competition Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Total number of teams. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
            num_teams: Teams per league. Defaults to 10.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.num_teams = min(num_teams, population_size)

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Soccer League Competition algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize teams (positions)
        teams = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(team) for team in teams])

        best_idx = np.argmin(fitness)
        best_solution = teams[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Sort teams by fitness
        sorted_indices = np.argsort(fitness)

        for iteration in range(self.max_iter):
            t = iteration / self.max_iter

            for i in range(self.population_size):
                # Select opponent (weighted toward better teams)
                weights = 1.0 / (np.arange(self.population_size) + 1)
                weights /= weights.sum()
                opponent_idx = np.random.choice(self.population_size, p=weights)

                # Match process
                if fitness[i] < fitness[opponent_idx]:
                    # Winner (team i) - improve slightly
                    r1 = np.random.random(self.dim)
                    new_position = teams[i] + r1 * (best_solution - teams[i]) * (1 - t)
                else:
                    # Loser (team i) - learn from opponent
                    r2 = np.random.random(self.dim)
                    new_position = teams[i] + r2 * (teams[opponent_idx] - teams[i])

                # Training phase (random improvement)
                if np.random.random() < 0.2:  # 20% training probability
                    r3 = np.random.uniform(-1, 1, self.dim)
                    training = (
                        r3 * (1 - t) * (self.upper_bound - self.lower_bound) * 0.1
                    )
                    new_position = new_position + training

                # Transfer window (swap dimensions with random team)
                if np.random.random() < 0.1:  # 10% transfer probability
                    j = np.random.randint(self.population_size)
                    dim_to_swap = np.random.randint(self.dim)
                    new_position[dim_to_swap] = teams[j][dim_to_swap]

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                new_fitness = self.func(new_position)

                # Update if improved
                if new_fitness < fitness[i]:
                    teams[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

            # Update rankings
            sorted_indices = np.argsort(fitness)

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = SoccerLeagueOptimizer(
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
