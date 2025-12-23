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

        >>> from opt.multi_objective.moead import MOEAD
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = MOEAD(
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
        >>> optimizer = MOEAD(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: objectives, lower_bound, upper_bound, dim, population_size, max_iter, n_neighbors

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
