"""Marine Predators Algorithm (MPA).

This module implements the Marine Predators Algorithm, a nature-inspired
metaheuristic based on the foraging strategy of ocean predators.

The algorithm mimics the Lévy and Brownian motion strategies used by marine
predators when hunting prey, with the choice of movement depending on the
velocity ratio between predator and prey.

Reference:
    Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
    Marine Predators Algorithm: A nature-inspired metaheuristic.
    Expert Systems with Applications, 152, 113377.
    DOI: 10.1016/j.eswa.2020.113377

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = MarinePredatorsOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=10,
    ...     population_size=30,
    ...     max_iter=500,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
    >>> isinstance(float(best_fitness), float)
    True

Attributes:
    func (Callable): The objective function to minimize.
    lower_bound (float): Lower bound of the search space.
    upper_bound (float): Upper bound of the search space.
    dim (int): Dimensionality of the search space.
    population_size (int): Number of prey in the population.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for Marine Predators Algorithm
_FADs_EFFECT_PROB = 0.2  # Fish Aggregating Devices effect probability
_FADs_CONSTRUCTION_THRESHOLD = 0.5  # Threshold for FADs construction vs destruction
_PHASE_TRANSITION_1 = 1 / 3  # First phase transition point
_PHASE_TRANSITION_2 = 2 / 3  # Second phase transition point
_LEVY_BETA = 1.5  # Lévy flight parameter


class MarinePredatorsOptimizer(AbstractOptimizer):
    r"""FIXME: [Algorithm Full Name] ([ACRONYM]) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | FIXME: [Full algorithm name]             |
        | Acronym           | FIXME: [SHORT]                           |
        | Year Introduced   | FIXME: [YYYY]                            |
        | Authors           | FIXME: [Last, First; ...]                |
        | Algorithm Class   | Swarm Intelligence |
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

        >>> from opt.swarm_intelligence.marine_predators_algorithm import MarinePredatorsOptimizer
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = MarinePredatorsOptimizer(
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
        >>> optimizer = MarinePredatorsOptimizer(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: func, lower_bound, upper_bound, dim, max_iter, seed, population_size, fads

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
        search() -> tuple[np.ndarray, float]:
            Execute optimization algorithm.

    Returns:
        tuple[np.ndarray, float]:
        Best solution found and its fitness value

    Raises:
        ValueError: If search space is invalid or function evaluation fails.

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
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 100,
        fads: float = _FADs_EFFECT_PROB,
    ) -> None:
        """Initialize the Marine Predators Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Problem dimensionality.
            max_iter: Maximum iterations.
            seed: Random seed.
            population_size: Number of prey.
            fads: Fish Aggregating Devices effect probability.
        """
        super().__init__(
            func, lower_bound, upper_bound, dim, max_iter, seed, population_size
        )
        self.fads = fads

    def _levy_flight(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Generate Lévy flight steps.

        Args:
            rng: Random number generator.
            size: Size of the step vector.

        Returns:
        Lévy flight step vector.
        """
        sigma = (
            math.gamma(1 + _LEVY_BETA)
            * np.sin(np.pi * _LEVY_BETA / 2)
            / (
                math.gamma((1 + _LEVY_BETA) / 2)
                * _LEVY_BETA
                * 2 ** ((_LEVY_BETA - 1) / 2)
            )
        ) ** (1 / _LEVY_BETA)

        u = rng.normal(0, sigma, size)
        v = rng.normal(0, 1, size)

        return u / (np.abs(v) ** (1 / _LEVY_BETA))

    def _brownian_motion(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Generate Brownian motion steps.

        Args:
            rng: Random number generator.
            size: Size of the step vector.

        Returns:
        Brownian motion step vector.
        """
        return rng.normal(0, 1, size)

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Marine Predators Algorithm.

        Returns:
        Tuple containing:
        - best_solution: The best solution found (numpy array).
        - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize prey population
        prey = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(p) for p in prey])

        # Find top predator (elite)
        best_idx = np.argmin(fitness)
        elite = prey[best_idx].copy()
        elite_fitness = fitness[best_idx]

        # Create Elite matrix (all rows are copies of elite)
        elite_matrix = np.tile(elite, (self.population_size, 1))

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Calculate CF (control factor)
            cf = (1 - iteration / self.max_iter) ** (2 * iteration / self.max_iter)

            # Determine phase
            progress = iteration / self.max_iter

            # Update each prey position
            for i in range(self.population_size):
                r = rng.random(self.dim)
                step_size = np.zeros(self.dim)

                if progress < _PHASE_TRANSITION_1:
                    # Phase 1: High velocity ratio (prey moves faster) - Brownian
                    rb = self._brownian_motion(rng, self.dim)
                    step_size = rb * (elite_matrix[i] - rb * prey[i])
                    prey[i] = prey[i] + 0.5 * r * step_size

                elif progress < _PHASE_TRANSITION_2:
                    # Phase 2: Unit velocity ratio - mixed exploration
                    if i < self.population_size // 2:
                        # First half: Lévy based on prey
                        rl = self._levy_flight(rng, self.dim)
                        step_size = rl * (elite_matrix[i] - rl * prey[i])
                        prey[i] = prey[i] + 0.5 * r * step_size
                    else:
                        # Second half: Brownian based on elite
                        rb = self._brownian_motion(rng, self.dim)
                        step_size = rb * (rb * elite_matrix[i] - prey[i])
                        prey[i] = elite_matrix[i] + 0.5 * cf * step_size

                else:
                    # Phase 3: Low velocity ratio (predator faster) - Lévy
                    rl = self._levy_flight(rng, self.dim)
                    step_size = rl * (rl * elite_matrix[i] - prey[i])
                    prey[i] = elite_matrix[i] + 0.5 * cf * step_size

                # Ensure bounds
                prey[i] = np.clip(prey[i], self.lower_bound, self.upper_bound)

            # FADs effect (Fish Aggregating Devices)
            if rng.random() < self.fads:
                # Eddy formation and FADs effect
                r = rng.random()
                u = np.ones((self.population_size, self.dim)) * (
                    rng.random((self.population_size, self.dim)) < self.fads
                )

                if r < _FADs_CONSTRUCTION_THRESHOLD:
                    # FADs construction
                    indices = rng.permutation(self.population_size)
                    prey = (
                        prey
                        + cf
                        * (
                            self.lower_bound
                            + rng.random((self.population_size, self.dim))
                            * (self.upper_bound - self.lower_bound)
                        )
                        * u
                    )
                else:
                    # FADs destruction
                    indices = rng.permutation(self.population_size)
                    prey = prey + (self.fads * (1 - r) + r) * (
                        prey[
                            indices[: self.population_size // 2].repeat(2)[
                                : self.population_size
                            ]
                        ]
                        - prey[
                            indices[self.population_size // 2 :].repeat(2)[
                                : self.population_size
                            ]
                        ]
                    )

                # Ensure bounds after FADs effect
                prey = np.clip(prey, self.lower_bound, self.upper_bound)

            # Update fitness
            fitness = np.array([self.func(p) for p in prey])

            # Update elite
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < elite_fitness:
                elite = prey[best_idx].copy()
                elite_fitness = fitness[best_idx]
                elite_matrix = np.tile(elite, (self.population_size, 1))

        return elite, elite_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(MarinePredatorsOptimizer)
