"""NSGA-II: Non-dominated Sorting Genetic Algorithm II.

This module implements the NSGA-II algorithm, one of the most popular and
highly-cited multi-objective evolutionary optimization algorithms.

NSGA-II uses fast non-dominated sorting and crowding distance assignment
to maintain a well-spread Pareto-optimal front while efficiently converging
to the true Pareto front.

Reference:
    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
    A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.
    IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
    DOI: 10.1109/4235.996017

Example:
    >>> def f1(x):
    ...     return sum(x**2)
    >>> def f2(x):
    ...     return sum((x - 2) ** 2)
    >>> optimizer = NSGAII(
    ...     objectives=[f1, f2],
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=10,
    ...     population_size=100,
    ...     max_iter=200,
    ... )
    >>> pareto_solutions, pareto_fitness = optimizer.search()
    >>> print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")

Attributes:
    objectives (list): List of objective functions to minimize.
    lower_bound (float): Lower bound of the search space.
    upper_bound (float): Upper bound of the search space.
    dim (int): Dimensionality of the search space.
    population_size (int): Number of individuals in the population.
    max_iter (int): Maximum number of generations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.multi_objective.abstract_multi_objective import AbstractMultiObjectiveOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


# Module-level constants for genetic operators
_CROSSOVER_PROBABILITY = 0.9
_MUTATION_PROBABILITY = 0.1
_TOURNAMENT_SIZE = 2
_SBX_DISTRIBUTION_INDEX = 20.0
_POLYNOMIAL_MUTATION_INDEX = 20.0
_CROSSOVER_DIMENSION_PROB = 0.5
_MUTATION_DIRECTION_PROB = 0.5
_CROSSOVER_EPSILON = 1e-14


class NSGAII(AbstractMultiObjectiveOptimizer):
    """NSGA-II Multi-Objective Optimizer.

    Non-dominated Sorting Genetic Algorithm II uses:
    - Fast non-dominated sorting for population ranking
    - Crowding distance for diversity preservation
    - Binary tournament selection based on rank and crowding
    - Simulated Binary Crossover (SBX) for offspring creation
    - Polynomial mutation for solution perturbation

    Attributes:
        crossover_prob (float): Probability of crossover.
        mutation_prob (float): Probability of mutation per dimension.
        tournament_size (int): Number of individuals in tournament selection.
        eta_c (float): Distribution index for SBX crossover.
        eta_m (float): Distribution index for polynomial mutation.

    Example:
        >>> from opt.multi_objective.nsga_ii import NSGAII
        >>> from opt.benchmark.functions import sphere, rosenbrock
        >>> optimizer = NSGAII(
        ...     objectives=[sphere, rosenbrock], dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> pareto_front, pareto_solutions = optimizer.search()
        >>> len(pareto_solutions) > 0  # Should find solutions
        True

    Example with single objective:
        >>> from opt.benchmark.functions import sphere
        >>> import numpy as np
        >>> optimizer = NSGAII(
        ...     objectives=[sphere], dim=2,
        ...     lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> pareto_front, _ = optimizer.search()
        >>> isinstance(pareto_front, np.ndarray)
        True
    """

    def __init__(
        self,
        objectives: Sequence[Callable[[np.ndarray], float]],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 200,
        seed: int | None = None,
        population_size: int = 100,
        crossover_prob: float = _CROSSOVER_PROBABILITY,
        mutation_prob: float | None = None,
        tournament_size: int = _TOURNAMENT_SIZE,
        eta_c: float = _SBX_DISTRIBUTION_INDEX,
        eta_m: float = _POLYNOMIAL_MUTATION_INDEX,
    ) -> None:
        """Initialize NSGA-II optimizer.

        Args:
            objectives: List of objective functions to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Problem dimensionality.
            max_iter: Maximum number of generations.
            seed: Random seed.
            population_size: Size of population.
            crossover_prob: Crossover probability.
            mutation_prob: Mutation probability (default: 1/dim).
            tournament_size: Tournament selection size.
            eta_c: SBX distribution index.
            eta_m: Polynomial mutation distribution index.
        """
        super().__init__(
            objectives, lower_bound, upper_bound, dim, max_iter, seed, population_size
        )
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob if mutation_prob else 1.0 / dim
        self.tournament_size = tournament_size
        self.eta_c = eta_c
        self.eta_m = eta_m

    def _initialize_population(self, rng: np.random.Generator) -> np.ndarray:
        """Initialize random population.

        Args:
            rng: Random number generator.

        Returns:
            Initial population array.
        """
        return rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

    def _tournament_selection(
        self,
        rng: np.random.Generator,
        population: np.ndarray,
        ranks: np.ndarray,
        crowding: np.ndarray,
    ) -> np.ndarray:
        """Select parent using binary tournament.

        Args:
            rng: Random number generator.
            population: Current population.
            ranks: Pareto ranks for each individual.
            crowding: Crowding distances for each individual.

        Returns:
            Selected parent.
        """
        candidates = rng.choice(len(population), self.tournament_size, replace=False)

        # Compare candidates: prefer lower rank, then higher crowding distance
        best = candidates[0]
        for candidate in candidates[1:]:
            if ranks[candidate] < ranks[best] or (
                ranks[candidate] == ranks[best] and crowding[candidate] > crowding[best]
            ):
                best = candidate

        return population[best].copy()

    def _sbx_crossover(
        self, rng: np.random.Generator, parent1: np.ndarray, parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX).

        Args:
            rng: Random number generator.
            parent1: First parent.
            parent2: Second parent.

        Returns:
            Tuple of two offspring.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        if rng.random() < self.crossover_prob:
            for i in range(self.dim):
                if (
                    rng.random() < _CROSSOVER_DIMENSION_PROB
                    and abs(parent1[i] - parent2[i]) > _CROSSOVER_EPSILON
                ):
                    y1 = min(parent1[i], parent2[i])
                    y2 = max(parent1[i], parent2[i])

                    beta = 1.0 + (2.0 * (y1 - self.lower_bound) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(self.eta_c + 1))
                    rand = rng.random()
                    betaq = (
                        (rand * alpha) ** (1.0 / (self.eta_c + 1))
                        if rand <= (1.0 / alpha)
                        else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1))
                    )

                    child1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                    beta = 1.0 + (2.0 * (self.upper_bound - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(self.eta_c + 1))
                    betaq = (
                        (rand * alpha) ** (1.0 / (self.eta_c + 1))
                        if rand <= (1.0 / alpha)
                        else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1))
                    )

                    child2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    # Ensure bounds
                    child1[i] = np.clip(child1[i], self.lower_bound, self.upper_bound)
                    child2[i] = np.clip(child2[i], self.lower_bound, self.upper_bound)

        return child1, child2

    def _polynomial_mutation(
        self, rng: np.random.Generator, individual: np.ndarray
    ) -> np.ndarray:
        """Polynomial mutation.

        Args:
            rng: Random number generator.
            individual: Individual to mutate.

        Returns:
            Mutated individual.
        """
        mutant = individual.copy()

        for i in range(self.dim):
            if rng.random() < self.mutation_prob:
                y = individual[i]
                delta1 = (y - self.lower_bound) / (self.upper_bound - self.lower_bound)
                delta2 = (self.upper_bound - y) / (self.upper_bound - self.lower_bound)

                rand = rng.random()
                if rand < _MUTATION_DIRECTION_PROB:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_m + 1))
                    deltaq = val ** (1.0 / (self.eta_m + 1)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (
                        xy ** (self.eta_m + 1)
                    )
                    deltaq = 1.0 - val ** (1.0 / (self.eta_m + 1))

                mutant[i] = y + deltaq * (self.upper_bound - self.lower_bound)
                mutant[i] = np.clip(mutant[i], self.lower_bound, self.upper_bound)

        return mutant

    def _assign_ranks_and_crowding(
        self, fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign Pareto ranks and crowding distances.

        Args:
            fitness: Fitness values for all individuals.

        Returns:
            Tuple of (ranks, crowding_distances) arrays.
        """
        fronts = self.fast_non_dominated_sort(fitness)
        n = len(fitness)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)

        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
            distances = self.crowding_distance(fitness, front)
            for i, idx in enumerate(front):
                crowding[idx] = distances[i]

        return ranks, crowding

    def search(self) -> tuple[np.ndarray, np.ndarray]:
        """Execute the NSGA-II algorithm.

        Returns:
            Tuple containing:
                - pareto_solutions: 2D array of Pareto-optimal solutions.
                - pareto_fitness: 2D array of objective values.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize population
        population = self._initialize_population(rng)
        fitness = self.evaluate_population(population)

        # Assign initial ranks and crowding
        ranks, crowding = self._assign_ranks_and_crowding(fitness)

        # Main generation loop
        for _ in range(self.max_iter):
            # Create offspring population
            offspring = []
            while len(offspring) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(rng, population, ranks, crowding)
                parent2 = self._tournament_selection(rng, population, ranks, crowding)

                # Crossover
                child1, child2 = self._sbx_crossover(rng, parent1, parent2)

                # Mutation
                child1 = self._polynomial_mutation(rng, child1)
                child2 = self._polynomial_mutation(rng, child2)

                offspring.extend([child1, child2])

            offspring = np.array(offspring[: self.population_size])
            offspring_fitness = self.evaluate_population(offspring)

            # Combine parent and offspring populations
            combined_pop = np.vstack([population, offspring])
            combined_fitness = np.vstack([fitness, offspring_fitness])

            # Sort and select next generation
            fronts = self.fast_non_dominated_sort(combined_fitness)

            new_population = []
            new_fitness = []

            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    # Add entire front
                    for idx in front:
                        new_population.append(combined_pop[idx])
                        new_fitness.append(combined_fitness[idx])
                else:
                    # Sort by crowding distance and add remaining
                    distances = self.crowding_distance(combined_fitness, front)
                    sorted_front = [front[i] for i in np.argsort(distances)[::-1]]
                    remaining = self.population_size - len(new_population)
                    for idx in sorted_front[:remaining]:
                        new_population.append(combined_pop[idx])
                        new_fitness.append(combined_fitness[idx])
                    break

            population = np.array(new_population)
            fitness = np.array(new_fitness)
            ranks, crowding = self._assign_ranks_and_crowding(fitness)

        # Extract Pareto front (rank 0 solutions)
        pareto_mask = ranks == 0
        return population[pareto_mask], fitness[pareto_mask]


if __name__ == "__main__":
    # ZDT1 test problem
    def zdt1_f1(x: np.ndarray) -> float:
        """ZDT1 first objective."""
        return x[0]

    def zdt1_f2(x: np.ndarray) -> float:
        """ZDT1 second objective."""
        n = len(x)
        g = 1 + 9 * np.sum(x[1:]) / (n - 1)
        return g * (1 - np.sqrt(x[0] / g))

    optimizer = NSGAII(
        objectives=[zdt1_f1, zdt1_f2],
        lower_bound=0.0,
        upper_bound=1.0,
        dim=10,
        population_size=100,
        max_iter=200,
    )
    pareto_solutions, pareto_fitness = optimizer.search()
    print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
    print(
        f"Objective range: f1=[{pareto_fitness[:, 0].min():.4f}, {pareto_fitness[:, 0].max():.4f}]"
    )
    print(
        f"                 f2=[{pareto_fitness[:, 1].min():.4f}, {pareto_fitness[:, 1].max():.4f}]"
    )
