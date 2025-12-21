"""Political Optimizer Algorithm.

This module implements the Political Optimizer, a social-inspired metaheuristic
algorithm based on political strategies and election processes.

The algorithm simulates political party behavior including constituency
allocation, party switching, and election campaigns.

Reference:
    Askari, Q., Younas, I., & Saeed, M. (2020).
    Political Optimizer: A novel socio-inspired meta-heuristic for global
    optimization.
    Knowledge-Based Systems, 195, 105709.
    DOI: 10.1016/j.knosys.2020.105709

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = PoliticalOptimizer(
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


class PoliticalOptimizer(AbstractOptimizer):
    """Political Optimizer algorithm.

    This algorithm simulates political behaviors:
    1. Constituency allocation - dividing search space
    2. Party switching - moving toward better parties
    3. Election campaign - exploitation phase

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of politicians.
        max_iter: Maximum number of iterations (elections).
        num_parties: Number of political parties.


    Example:
        >>> from opt.social_inspired.political_optimizer import PoliticalOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = PoliticalOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = PoliticalOptimizer(
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
        num_parties: int = 5,
    ) -> None:
        """Initialize Political Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of politicians. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
            num_parties: Number of parties. Defaults to 5.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.num_parties = min(num_parties, population_size)

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Political Optimizer.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population (politicians)
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])

        # Assign politicians to parties
        party_assignment = np.random.randint(0, self.num_parties, self.population_size)

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            t = iteration / self.max_iter

            # Find party leaders (best in each party)
            party_leaders = np.zeros((self.num_parties, self.dim))
            party_leader_fitness = np.full(self.num_parties, np.inf)

            for p in range(self.num_parties):
                party_members = np.where(party_assignment == p)[0]
                if len(party_members) > 0:
                    best_member = party_members[np.argmin(fitness[party_members])]
                    party_leaders[p] = population[best_member]
                    party_leader_fitness[p] = fitness[best_member]

            for i in range(self.population_size):
                current_party = party_assignment[i]
                leader = party_leaders[current_party]

                r = np.random.random()

                if r < 0.5:
                    # Constituency allocation (exploration)
                    # Politicians explore their constituency
                    r1 = np.random.random(self.dim)
                    r2 = np.random.random()

                    new_position = population[i] + r1 * (leader - r2 * population[i])

                else:
                    # Election campaign (exploitation)
                    # Move toward party leader or switch parties
                    if np.random.random() < 0.3 * (1 - t):  # Party switching
                        # Switch to a better party
                        better_parties = np.where(party_leader_fitness < fitness[i])[0]
                        if len(better_parties) > 0:
                            new_party = np.random.choice(better_parties)
                            party_assignment[i] = new_party
                            leader = party_leaders[new_party]

                    r3 = np.random.random(self.dim)
                    r4 = np.random.random()

                    # Campaign toward best solution
                    new_position = (
                        population[i]
                        + r3 * (best_solution - population[i])
                        + r4 * (1 - t) * (leader - population[i])
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
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

    run_demo(PoliticalOptimizer)
