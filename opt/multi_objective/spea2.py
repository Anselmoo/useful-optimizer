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
    r"""FIXME: [Algorithm Full Name] ([ACRONYM]) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | FIXME: [Full algorithm name]             |
        | Acronym           | FIXME: [SHORT]                           |
        | Year Introduced   | FIXME: [YYYY]                            |
        | Authors           | FIXME: [Last, First; ...]                |
        | Algorithm Class   | Multi Objective |
        | Complexity        | FIXME: O([expression])                   |
        | Properties        | FIXME: [Population-based, ...]           |
        | Implementation    | Python 3.10+                             |
        | COCO Compatible   | Yes                                      |

    Mathematical Formulation:
        FIXME: Core update equation:

            $$
            x_{t+1} = x_t + v_t
            $$

        where:
            - $x_t$ is the position at iteration $t$
            - $v_t$ is the velocity/step at iteration $t$
            - FIXME: Additional variable definitions...

        Constraint handling:
            - **Boundary conditions**: FIXME: [clamping/reflection/periodic]
            - **Feasibility enforcement**: FIXME: [description]

    Hyperparameters:
        | Parameter              | Default | BBOB Recommended | Description                    |
        |------------------------|---------|------------------|--------------------------------|
        | population_size        | 100     | 10*dim           | Number of individuals          |
        | max_iter               | 1000    | 10000            | Maximum iterations             |
        | FIXME: [param_name]    | [val]   | [bbob_val]       | [description]                  |

        **Sensitivity Analysis**:
            - FIXME: `[param_name]`: **[High/Medium/Low]** impact on convergence
            - Recommended tuning ranges: FIXME: $\text{[param]} \in [\text{min}, \text{max}]$

    COCO/BBOB Benchmark Settings:
        **Search Space**:
            - Dimensions tested: `2, 3, 5, 10, 20, 40`
            - Bounds: Function-specific (typically `[-5, 5]` or `[-100, 100]`)
            - Instances: **15** per function (BBOB standard)

        **Evaluation Budget**:
            - Budget: $\text{dim} \times 10000$ function evaluations
            - Independent runs: **15** (for statistical significance)
            - Seeds: `0-14` (reproducibility requirement)

        **Performance Metrics**:
            - Target precision: `1e-8` (BBOB default)
            - Success rate at precision thresholds: `[1e-8, 1e-6, 1e-4, 1e-2]`
            - Expected Running Time (ERT) tracking

    Example:
        Basic usage with BBOB benchmark function:

        >>> from opt.multi_objective.spea2 import SPEA2
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SPEA2(
        ...     func=shifted_ackley,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     dim=2,
        ...     max_iter=100,
        ...     seed=42,  # Required for reproducibility
        ... )
        >>> solution, fitness = optimizer.search()
        >>> isinstance(fitness, float) and fitness >= 0
        True

        COCO benchmark example:

        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SPEA2(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: objectives, lower_bound, upper_bound, dim, max_iter, population_size, archive_size

        Common parameters (adjust based on actual signature):
        func (Callable[[ndarray], float]): Objective function to minimize. Must accept
            numpy array and return scalar. BBOB functions available in
            `opt.benchmark.functions`.
        lower_bound (float): Lower bound of search space. BBOB typical: -5
            (most functions).
        upper_bound (float): Upper bound of search space. BBOB typical: 5
            (most functions).
        dim (int): Problem dimensionality. BBOB standard dimensions: 2, 3, 5, 10, 20, 40.
        max_iter (int, optional): Maximum iterations. BBOB recommendation: 10000 for
            complete evaluation. Defaults to 1000.
        seed (int | None, optional): Random seed for reproducibility. BBOB requires
            seeds 0-14 for 15 runs. If None, generates random seed. Defaults to None.
        population_size (int, optional): Population size. BBOB recommendation: 10*dim
            for population-based methods. Defaults to 100. (Only for population-based
            algorithms)
        track_history (bool, optional): Enable convergence history tracking for BBOB
            post-processing. Defaults to False.
        FIXME: [algorithm_specific_params] ([type], optional): FIXME: Document any
            algorithm-specific parameters not listed above. Defaults to [value].

    Attributes:
        func (Callable[[ndarray], float]): The objective function being optimized.
        lower_bound (float): Lower search space boundary.
        upper_bound (float): Upper search space boundary.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum number of iterations.
        seed (int): **REQUIRED** Random seed for reproducibility (BBOB compliance).
        population_size (int): Number of individuals in population.
        track_history (bool): Whether convergence history is tracked.
        history (dict[str, list]): Optimization history if track_history=True. Contains:
            - 'best_fitness': list[float] - Best fitness per iteration
            - 'best_solution': list[ndarray] - Best solution per iteration
            - 'population_fitness': list[ndarray] - All fitness values
            - 'population': list[ndarray] - All solutions
        FIXME: [algorithm_specific_attrs] ([type]): FIXME: [Description]

    Methods:
        search() -> tuple[ndarray, ndarray]:
            Execute optimization algorithm.

    Returns:
        tuple[ndarray, ndarray]: Pareto-optimal solutions and their fitness values

    Raises:
        ValueError:
        If search space is invalid or function evaluation fails.

    Notes:
        - Modifies self.history if track_history=True
        - Uses self.seed for all random number generation
        - BBOB: Returns final best solution after max_iter or convergence

    References:
        FIXME: [1] Author1, A., Author2, B. (YEAR). "Algorithm Name: Description."
        _Journal Name_, Volume(Issue), Pages.
        https://doi.org/10.xxxx/xxxxx

        [2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
            "COCO: A platform for comparing continuous optimizers in a black-box setting."
            _Optimization Methods and Software_, 36(1), 114-144.
            https://doi.org/10.1080/10556788.2020.1808977

        **COCO Data Archive**:
            - Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
            - FIXME: Algorithm data: [URL to algorithm-specific COCO results if available]
            - Code repository: https://github.com/Anselmoo/useful-optimizer

        **Implementation**:
            - FIXME: Original paper code: [URL if different from this implementation]
            - This implementation: Based on [1] with modifications for BBOB compliance

    See Also:
        FIXME: [RelatedAlgorithm1]: Similar algorithm with [key difference]
            BBOB Comparison: [Brief performance notes on sphere/rosenbrock/ackley]

        FIXME: [RelatedAlgorithm2]: [Relationship description]
            BBOB Comparison: Generally [faster/slower/more robust] on [function classes]

        AbstractOptimizer: Base class for all optimizers
        opt.benchmark.functions: BBOB-compatible test functions

        Related BBOB Algorithm Classes:
            - Evolutionary: GeneticAlgorithm, DifferentialEvolution
            - Swarm: ParticleSwarm, AntColony
            - Gradient: AdamW, SGDMomentum

    Notes:
        **Computational Complexity**:
        - Time per iteration: FIXME: $O(\text{[expression]})$
        - Space complexity: FIXME: $O(\text{[expression]})$
        - BBOB budget usage: FIXME: _[Typical percentage of dim*10000 budget needed]_

        **BBOB Performance Characteristics**:
            - **Best function classes**: FIXME: [Unimodal/Multimodal/Ill-conditioned/...]
            - **Weak function classes**: FIXME: [Function types where algorithm struggles]
            - Typical success rate at 1e-8 precision: FIXME: **[X]%** (dim=5)
            - Expected Running Time (ERT): FIXME: [Comparative notes vs other algorithms]

        **Convergence Properties**:
            - Convergence rate: FIXME: [Linear/Quadratic/Exponential]
            - Local vs Global: FIXME: [Tendency for local/global optima]
            - Premature convergence risk: FIXME: **[High/Medium/Low]**

        **Reproducibility**:
            - **Deterministic**: FIXME: [Yes/No] - Same seed guarantees same results
            - **BBOB compliance**: seed parameter required for 15 independent runs
            - Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
            - RNG usage: `numpy.random.default_rng(self.seed)` throughout

        **Implementation Details**:
            - Parallelization: FIXME: [Not supported/Supported via `[method]`]
            - Constraint handling: FIXME: [Clamping to bounds/Penalty/Repair]
            - Numerical stability: FIXME: [Considerations for floating-point arithmetic]

        **Known Limitations**:
            - FIXME: [Any known issues or limitations specific to this implementation]
            - FIXME: BBOB known issues: [Any BBOB-specific challenges]

        **Version History**:
            - v0.1.0: Initial implementation
            - FIXME: [vX.X.X]: [Changes relevant to BBOB compliance]
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
        """Test objective f1 for the SPEA2 example."""
        return x[0] ** 2 + x[1] ** 2

    def f2(x: np.ndarray) -> float:
        """Test objective f2 for the SPEA2 example."""
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
