"""MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition).

This module implements MOEA/D, a highly influential multi-objective
optimization algorithm that decomposes a multi-objective problem into
scalar subproblems.

Reference:
    Zhang, Q., & Li, H. (2007).
    MOEA/D: A multiobjective evolutionary algorithm based on decomposition.
    IEEE Transactions on Evolutionary Computation, 11(6), 712-731.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.multi_objective.abstract_multi_objective import AbstractMultiObjectiveOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for MOEA/D algorithm
_CROSSOVER_RATE = 1.0  # SBX crossover probability
_MUTATION_RATE = 0.1  # Polynomial mutation probability
_SBX_ETA = 20  # SBX distribution index
_PM_ETA = 20  # Polynomial mutation distribution index
_NEIGHBOR_SELECTION_PROB = 0.9  # Probability of selecting from neighborhood
_MAX_REPLACE = 2  # Maximum number of solutions replaced by each offspring
_CROSSOVER_DIM_PROB = 0.5  # Probability to apply crossover to each dimension
_EPSILON = 1e-14  # Small value to avoid division by zero
_MUTATION_MIDPOINT = 0.5  # Midpoint for mutation direction
_BI_OBJECTIVE = 2  # Number of objectives for bi-objective problems


class MOEAD(AbstractMultiObjectiveOptimizer):
    """MOEA/D implementation.

    MOEA/D decomposes a multi-objective problem into N scalar
    subproblems using weight vectors and optimizes them simultaneously
    using neighborhood relations.

    Attributes:
        objectives: List of objective functions to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        population_size: Number of weight vectors/subproblems.
        max_iter: Maximum number of iterations.
        n_neighbors: Number of neighbors for each subproblem.
    """

    def __init__(
        self,
        objectives: list[Callable[[np.ndarray], float]],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 100,
        max_iter: int = 300,
        n_neighbors: int = 20,
    ) -> None:
        """Initialize the MOEA/D optimizer.

        Args:
            objectives: List of objective functions to minimize.
            lower_bound: Lower bound for all dimensions.
            upper_bound: Upper bound for all dimensions.
            dim: Number of dimensions.
            population_size: Number of subproblems (weight vectors).
            max_iter: Maximum iterations.
            n_neighbors: Number of neighbors for each subproblem.
        """
        super().__init__(objectives, lower_bound, upper_bound, dim)
        self.population_size = population_size
        self.max_iter = max_iter
        self.n_neighbors = min(n_neighbors, population_size)
        self.n_objectives = len(objectives)

    def _generate_weight_vectors(self) -> np.ndarray:
        """Generate uniformly distributed weight vectors.

        Returns:
            Weight vectors of shape (population_size, n_objectives).
        """
        if self.n_objectives == _BI_OBJECTIVE:
            # Simple uniform distribution for 2 objectives
            weights = np.zeros((self.population_size, _BI_OBJECTIVE))
            for i in range(self.population_size):
                w1 = (
                    i / (self.population_size - 1)
                    if self.population_size > 1
                    else _CROSSOVER_DIM_PROB
                )
                weights[i] = [w1, 1 - w1]
            return weights
        # Random weights normalized to sum to 1
        weights = np.random.rand(self.population_size, self.n_objectives)
        return weights / weights.sum(axis=1, keepdims=True)

    def _compute_neighbors(self, weights: np.ndarray) -> np.ndarray:
        """Compute neighborhood based on weight vector distances.

        Args:
            weights: Weight vectors.

        Returns:
            Neighborhood indices for each subproblem.
        """
        distances = np.zeros((self.population_size, self.population_size))
        for i in range(self.population_size):
            for j in range(self.population_size):
                distances[i, j] = np.linalg.norm(weights[i] - weights[j])

        return np.argsort(distances, axis=1)[:, : self.n_neighbors]

    def _tchebycheff(
        self, x_fitness: np.ndarray, weight: np.ndarray, z_star: np.ndarray
    ) -> float:
        """Compute Tchebycheff aggregation function.

        Args:
            x_fitness: Fitness values for solution.
            weight: Weight vector.
            z_star: Reference point (ideal point).

        Returns:
            Aggregated scalar value.
        """
        # Add small constant to avoid division by zero
        weight_adj = np.maximum(weight, 1e-6)
        return np.max(weight_adj * np.abs(x_fitness - z_star))

    def _sbx_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX).

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            Two offspring.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        if np.random.rand() > _CROSSOVER_RATE:
            return child1, child2

        for i in range(self.dim):
            if np.random.rand() > _CROSSOVER_DIM_PROB:
                continue

            if np.abs(parent1[i] - parent2[i]) < _EPSILON:
                continue

            y1 = min(parent1[i], parent2[i])
            y2 = max(parent1[i], parent2[i])

            rand = np.random.rand()

            # Calculate beta
            beta_l = 1 + (2 * (y1 - self.lower_bound) / (y2 - y1 + _EPSILON))
            beta_r = 1 + (2 * (self.upper_bound - y2) / (y2 - y1 + _EPSILON))

            alpha_l = 2 - beta_l ** (-(_SBX_ETA + 1))
            alpha_r = 2 - beta_r ** (-(_SBX_ETA + 1))

            if rand <= 1 / alpha_l:
                betaq_l = (rand * alpha_l) ** (1 / (_SBX_ETA + 1))
            else:
                betaq_l = (1 / (2 - rand * alpha_l)) ** (1 / (_SBX_ETA + 1))

            if rand <= 1 / alpha_r:
                betaq_r = (rand * alpha_r) ** (1 / (_SBX_ETA + 1))
            else:
                betaq_r = (1 / (2 - rand * alpha_r)) ** (1 / (_SBX_ETA + 1))

            child1[i] = 0.5 * ((y1 + y2) - betaq_l * (y2 - y1))
            child2[i] = 0.5 * ((y1 + y2) + betaq_r * (y2 - y1))

        child1 = np.clip(child1, self.lower_bound, self.upper_bound)
        child2 = np.clip(child2, self.lower_bound, self.upper_bound)

        return child1, child2

    def _polynomial_mutation(self, x: np.ndarray) -> np.ndarray:
        """Polynomial mutation.

        Args:
            x: Solution to mutate.

        Returns:
            Mutated solution.
        """
        y = x.copy()

        for i in range(self.dim):
            if np.random.rand() > _MUTATION_RATE:
                continue

            delta1 = (y[i] - self.lower_bound) / (self.upper_bound - self.lower_bound)
            delta2 = (self.upper_bound - y[i]) / (self.upper_bound - self.lower_bound)

            rand = np.random.rand()

            if rand < _MUTATION_MIDPOINT:
                xy = 1 - delta1
                val = 2 * rand + (1 - 2 * rand) * (xy ** (_PM_ETA + 1))
                deltaq = val ** (1 / (_PM_ETA + 1)) - 1
            else:
                xy = 1 - delta2
                val = 2 * (1 - rand) + 2 * (rand - _MUTATION_MIDPOINT) * (
                    xy ** (_PM_ETA + 1)
                )
                deltaq = 1 - val ** (1 / (_PM_ETA + 1))

            y[i] = y[i] + deltaq * (self.upper_bound - self.lower_bound)

        return np.clip(y, self.lower_bound, self.upper_bound)

    def search(self) -> tuple[np.ndarray, np.ndarray]:
        """Execute the optimization algorithm.

        Returns:
            Tuple of (pareto_front, pareto_set).
        """
        # Generate weight vectors
        weights = self._generate_weight_vectors()

        # Compute neighbors
        neighbors = self._compute_neighbors(weights)

        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness for all objectives
        fitness = np.array(
            [[func(ind) for func in self.objectives] for ind in population]
        )

        # Initialize reference point (ideal point)
        z_star = np.min(fitness, axis=0)

        # External archive for non-dominated solutions
        archive_solutions: list[np.ndarray] = []
        archive_fitness: list[np.ndarray] = []

        for _ in range(self.max_iter):
            for i in range(self.population_size):
                # Select mating pool
                if np.random.rand() < _NEIGHBOR_SELECTION_PROB:
                    mating_pool = neighbors[i]
                else:
                    mating_pool = np.arange(self.population_size)

                # Select parents
                parents_idx = np.random.choice(mating_pool, 2, replace=False)

                # Crossover
                child1, _ = self._sbx_crossover(
                    population[parents_idx[0]], population[parents_idx[1]]
                )

                # Mutation
                offspring = self._polynomial_mutation(child1)

                # Evaluate offspring
                offspring_fitness = np.array(
                    [func(offspring) for func in self.objectives]
                )

                # Update reference point
                z_star = np.minimum(z_star, offspring_fitness)

                # Update neighbors
                replace_count = 0
                indices = (
                    mating_pool
                    if np.random.rand() < _NEIGHBOR_SELECTION_PROB
                    else np.arange(self.population_size)
                )
                np.random.shuffle(indices)

                for j in indices:
                    if replace_count >= _MAX_REPLACE:
                        break

                    old_te = self._tchebycheff(fitness[j], weights[j], z_star)
                    new_te = self._tchebycheff(offspring_fitness, weights[j], z_star)

                    if new_te < old_te:
                        population[j] = offspring.copy()
                        fitness[j] = offspring_fitness.copy()
                        replace_count += 1

                # Update archive with non-dominated solutions
                is_dominated = False
                to_remove: list[int] = []

                for k, arch_fit in enumerate(archive_fitness):
                    if self.dominates(arch_fit, offspring_fitness):
                        is_dominated = True
                        break
                    if self.dominates(offspring_fitness, arch_fit):
                        to_remove.append(k)

                if not is_dominated:
                    for k in reversed(to_remove):
                        del archive_solutions[k]
                        del archive_fitness[k]
                    archive_solutions.append(offspring.copy())
                    archive_fitness.append(offspring_fitness.copy())

        # Return Pareto front from archive
        if archive_solutions:
            pareto_set = np.array(archive_solutions)
            pareto_front = np.array(archive_fitness)
        else:
            # Fall back to population non-dominated solutions
            pareto_set, pareto_front = self._extract_pareto_front(population, fitness)

        return pareto_front, pareto_set

    def _extract_pareto_front(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract non-dominated solutions from population.

        Args:
            population: Current population.
            fitness: Fitness values.

        Returns:
            Tuple of (pareto_front, pareto_set).
        """
        n = len(population)
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            for j in range(n):
                if i != j and self.dominates(fitness[j], fitness[i]):
                    is_dominated[i] = True
                    break

        pareto_indices = ~is_dominated
        return population[pareto_indices], fitness[pareto_indices]


if __name__ == "__main__":

    def zdt1_f1(x: np.ndarray) -> float:
        """ZDT1 first objective function."""
        return float(x[0])

    def zdt1_f2(x: np.ndarray) -> float:
        """ZDT1 second objective function."""
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        h = 1 - np.sqrt(x[0] / g)
        return float(g * h)

    optimizer = MOEAD(
        objectives=[zdt1_f1, zdt1_f2],
        lower_bound=0.0,
        upper_bound=1.0,
        dim=10,
        population_size=50,
        max_iter=100,
    )
    pareto_front, pareto_set = optimizer.search()
    print(f"Found {len(pareto_front)} Pareto-optimal solutions")
    if len(pareto_front) > 0:
        print(f"Sample Pareto front point: {pareto_front[0]}")
