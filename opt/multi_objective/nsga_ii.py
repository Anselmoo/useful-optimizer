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
    >>> from opt.multi_objective.nsga_ii import NSGAII
    >>> import numpy as np
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
    ...     max_iter=10,
    ... )
    >>> pareto_solutions, pareto_fitness = optimizer.search()
    >>> len(pareto_solutions) > 0
    True

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

        >>> from opt.multi_objective.nsga_ii import NSGAII
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = NSGAII(
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
        >>> optimizer = NSGAII(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: objectives, lower_bound, upper_bound, dim, max_iter, seed, population_size, crossover_prob, mutation_prob, tournament_size, eta_c, eta_m

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
