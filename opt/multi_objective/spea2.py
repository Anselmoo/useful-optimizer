"""SPEA2 (Strength Pareto Evolutionary Algorithm 2) implementation.

This module implements SPEA2, an improved version of the Strength Pareto
Evolutionary Algorithm for multi-objective optimization.

Reference:
    Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the
    strength pareto evolutionary algorithm. TIK-Report 103, ETH Zurich.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.multi_objective.abstract_multi_objective import AbstractMultiObjectiveOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_CROSSOVER_RATE = 0.9  # SBX crossover probability
_MUTATION_RATE = 0.1  # Polynomial mutation probability
_SBX_ETA = 15  # SBX distribution index
_PM_ETA = 20  # Polynomial mutation distribution index
_CROSSOVER_DIM_PROB = 0.5  # Per-dimension crossover probability
_MUTATION_MIDPOINT = 0.5  # Mutation direction threshold
_K_NEIGHBOR = 1  # k-th nearest neighbor for density estimation


class SPEA2(AbstractMultiObjectiveOptimizer):
    """SPEA2 multi-objective optimizer.

    SPEA2 improves upon SPEA with:
    - Fine-grained fitness assignment using strength and raw fitness
    - Density estimation using k-th nearest neighbor
    - Archive truncation based on clustering

    Attributes:
        objectives: List of objective functions to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Size of the population.
        archive_size: Size of the external archive.


    Example:
        >>> from opt.multi_objective.spea2 import SPEA2
        >>> from opt.benchmark.functions import sphere
        >>> import numpy as np
        >>> optimizer = SPEA2(
        ...     objectives=[sphere], dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10
        ... )
        >>> result = optimizer.search()
        >>> isinstance(result, tuple)
        True
    """

    def __init__(
        self,
        objectives: list[Callable[[np.ndarray], float]],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        population_size: int = 100,
        archive_size: int = 100,
    ) -> None:
        """Initialize SPEA2.

        Args:
            objectives: List of objective functions to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Size of the population.
            archive_size: Size of the external archive.
        """
        super().__init__(objectives, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.archive_size = archive_size

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (all objectives minimized).

        Args:
            obj1: First objective vector.
            obj2: Second objective vector.

        Returns:
            True if obj1 Pareto-dominates obj2.
        """
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _calculate_strength(self, objectives_values: np.ndarray) -> np.ndarray:
        """Calculate strength values for all individuals.

        Strength of an individual = number of solutions it dominates.

        Args:
            objectives_values: Array of objective values (n_solutions x n_objectives).

        Returns:
            Array of strength values.
        """
        n = len(objectives_values)
        strength = np.zeros(n)

        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(
                    objectives_values[i], objectives_values[j]
                ):
                    strength[i] += 1

        return strength

    def _calculate_raw_fitness(
        self, objectives_values: np.ndarray, strength: np.ndarray
    ) -> np.ndarray:
        """Calculate raw fitness values.

        Raw fitness = sum of strengths of all dominators.

        Args:
            objectives_values: Array of objective values.
            strength: Array of strength values.

        Returns:
            Array of raw fitness values.
        """
        n = len(objectives_values)
        raw_fitness = np.zeros(n)

        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(
                    objectives_values[j], objectives_values[i]
                ):
                    raw_fitness[i] += strength[j]

        return raw_fitness

    def _calculate_density(self, objectives_values: np.ndarray) -> np.ndarray:
        """Calculate density values using k-th nearest neighbor.

        Args:
            objectives_values: Array of objective values.

        Returns:
            Array of density values.
        """
        n = len(objectives_values)
        density = np.zeros(n)

        # Calculate pairwise distances
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(objectives_values[i] - objectives_values[j])
                    distances.append(dist)

            distances.sort()
            k = min(_K_NEIGHBOR, len(distances))
            if k > 0:
                sigma_k = distances[k - 1]
                density[i] = 1.0 / (sigma_k + 2.0)

        return density

    def _sbx_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX).

        Args:
            parent1: First parent solution.
            parent2: Second parent solution.

        Returns:
            Tuple of two offspring solutions.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        if np.random.rand() > _CROSSOVER_RATE:
            return child1, child2

        for i in range(self.dim):
            if np.random.rand() > _CROSSOVER_DIM_PROB:
                continue

            y1 = min(parent1[i], parent2[i])
            y2 = max(parent1[i], parent2[i])

            if abs(y2 - y1) < 1e-14:
                continue

            rand = np.random.rand()

            # Calculate beta
            beta = 1.0 + (2.0 * (y1 - self.lower_bound) / (y2 - y1))
            alpha = 2.0 - beta ** (-(_SBX_ETA + 1))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (_SBX_ETA + 1))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (_SBX_ETA + 1))

            child1[i] = _CROSSOVER_DIM_PROB * ((y1 + y2) - betaq * (y2 - y1))

            beta = 1.0 + (2.0 * (self.upper_bound - y2) / (y2 - y1))
            alpha = 2.0 - beta ** (-(_SBX_ETA + 1))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (_SBX_ETA + 1))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (_SBX_ETA + 1))

            child2[i] = _CROSSOVER_DIM_PROB * ((y1 + y2) + betaq * (y2 - y1))

        return child1, child2

    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation.

        Args:
            individual: Solution to mutate.

        Returns:
            Mutated solution.
        """
        y = individual.copy()

        for i in range(self.dim):
            if np.random.rand() > _MUTATION_RATE:
                continue

            delta1 = (y[i] - self.lower_bound) / (self.upper_bound - self.lower_bound)
            delta2 = (self.upper_bound - y[i]) / (self.upper_bound - self.lower_bound)

            rand = np.random.rand()

            if rand < _MUTATION_MIDPOINT:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (_PM_ETA + 1))
                deltaq = val ** (1.0 / (_PM_ETA + 1)) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - _MUTATION_MIDPOINT) * (
                    xy ** (_PM_ETA + 1)
                )
                deltaq = 1.0 - val ** (1.0 / (_PM_ETA + 1))

            y[i] += deltaq * (self.upper_bound - self.lower_bound)

        return y

    def _environmental_selection(
        self, combined_pop: np.ndarray, combined_obj: np.ndarray, fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Environmental selection to form the next archive.

        Args:
            combined_pop: Combined population array.
            combined_obj: Combined objective values.
            fitness: Fitness values for selection.

        Returns:
            Tuple of (selected_population, selected_objectives).
        """
        # Get non-dominated individuals (fitness < 1)
        non_dominated_mask = fitness < 1
        non_dominated_indices = np.where(non_dominated_mask)[0]

        if len(non_dominated_indices) <= self.archive_size:
            # If not enough non-dominated, fill with best dominated
            if len(non_dominated_indices) < self.archive_size:
                dominated_indices = np.where(~non_dominated_mask)[0]
                dominated_fitness = fitness[dominated_indices]
                sorted_dominated = dominated_indices[np.argsort(dominated_fitness)]

                needed = self.archive_size - len(non_dominated_indices)
                selected_indices = np.concatenate(
                    [non_dominated_indices, sorted_dominated[:needed]]
                )
            else:
                selected_indices = non_dominated_indices
        else:
            # Truncate using clustering
            selected_indices = self._truncate_archive(
                non_dominated_indices, combined_obj
            )

        return combined_pop[selected_indices], combined_obj[selected_indices]

    def _truncate_archive(
        self, indices: np.ndarray, objectives_values: np.ndarray
    ) -> np.ndarray:
        """Truncate archive using nearest neighbor clustering.

        Args:
            indices: Indices of non-dominated solutions.
            objectives_values: Objective values.

        Returns:
            Indices of selected solutions.
        """
        selected = list(indices)

        while len(selected) > self.archive_size:
            # Calculate all pairwise distances
            obj_selected = objectives_values[selected]
            n = len(selected)
            distances = np.full((n, n), np.inf)

            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(obj_selected[i] - obj_selected[j])
                    distances[i, j] = dist
                    distances[j, i] = dist

            # Find the individual with minimum distance to nearest neighbor
            min_distances = np.min(distances, axis=1)
            remove_idx = np.argmin(min_distances)

            selected.pop(remove_idx)

        return np.array(selected)

    def _binary_tournament(self, fitness: np.ndarray) -> int:
        """Binary tournament selection.

        Args:
            fitness: Array of fitness values.

        Returns:
            Index of selected individual.
        """
        idx1 = np.random.randint(len(fitness))
        idx2 = np.random.randint(len(fitness))

        if fitness[idx1] < fitness[idx2]:
            return idx1
        return idx2

    def search(self) -> tuple[np.ndarray, np.ndarray]:
        """Execute SPEA2.

        Returns:
            Tuple of (pareto_front_solutions, pareto_front_objectives).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate objectives
        objectives_values = np.array(
            [[obj(ind) for obj in self.objectives] for ind in population]
        )

        # Initialize archive
        archive = np.empty((0, self.dim))
        archive_obj = np.empty((0, len(self.objectives)))

        # Main loop
        for _ in range(self.max_iter):
            # Combine population and archive
            combined_pop = (
                np.vstack([population, archive]) if len(archive) > 0 else population
            )
            combined_obj = (
                np.vstack([objectives_values, archive_obj])
                if len(archive_obj) > 0
                else objectives_values
            )

            # Calculate fitness
            strength = self._calculate_strength(combined_obj)
            raw_fitness = self._calculate_raw_fitness(combined_obj, strength)
            density = self._calculate_density(combined_obj)
            fitness = raw_fitness + density

            # Environmental selection
            archive, archive_obj = self._environmental_selection(
                combined_pop, combined_obj, fitness
            )

            # Mating selection and variation
            archive_fitness = np.zeros(
                len(archive)
            )  # Archived solutions have fitness < 1
            offspring = []

            while len(offspring) < self.population_size:
                # Select parents
                p1_idx = self._binary_tournament(archive_fitness)
                p2_idx = self._binary_tournament(archive_fitness)
                while p2_idx == p1_idx:
                    p2_idx = self._binary_tournament(archive_fitness)

                # Crossover
                child1, child2 = self._sbx_crossover(archive[p1_idx], archive[p2_idx])

                # Mutation
                child1 = self._polynomial_mutation(child1)
                child2 = self._polynomial_mutation(child2)

                # Boundary handling
                child1 = np.clip(child1, self.lower_bound, self.upper_bound)
                child2 = np.clip(child2, self.lower_bound, self.upper_bound)

                offspring.extend([child1, child2])

            population = np.array(offspring[: self.population_size])
            objectives_values = np.array(
                [[obj(ind) for obj in self.objectives] for ind in population]
            )

        return archive, archive_obj


if __name__ == "__main__":
    # Test with simple bi-objective problem
    def f1(x: np.ndarray) -> float:
        return x[0] ** 2 + x[1] ** 2

    def f2(x: np.ndarray) -> float:
        return (x[0] - 1) ** 2 + (x[1] - 1) ** 2

    optimizer = SPEA2(
        objectives=[f1, f2],
        lower_bound=-2,
        upper_bound=2,
        dim=2,
        max_iter=50,
        population_size=50,
        archive_size=50,
    )
    pareto_solutions, pareto_objectives = optimizer.search()
    print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
    print(
        f"Objective ranges: f1=[{pareto_objectives[:, 0].min():.4f}, "
        f"{pareto_objectives[:, 0].max():.4f}], "
        f"f2=[{pareto_objectives[:, 1].min():.4f}, "
        f"{pareto_objectives[:, 1].max():.4f}]"
    )
