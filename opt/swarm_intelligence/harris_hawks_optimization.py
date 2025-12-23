"""Harris Hawks Optimization (HHO) Algorithm.

This module implements the Harris Hawks Optimization algorithm, a population-based
metaheuristic inspired by the cooperative hunting behavior of Harris hawks in nature.

The algorithm simulates the surprise pounce (or seven kills) strategy where
hawks cooperate to catch prey. It includes exploration and exploitation phases
with different attacking strategies based on the escaping energy of prey.

Reference:
    Heidari, A.A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M., & Chen, H.
    (2019). Harris hawks optimization: Algorithm and applications.
    Future Generation Computer Systems, 97, 849-872.
    DOI: 10.1016/j.future.2019.02.028

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = HarrisHawksOptimizer(
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
    population_size (int): Number of hawks in the population.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

import math

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


# Algorithm-specific constants (from original paper)
_EXPLORATION_THRESHOLD = 1.0  # |E| >= 1 triggers exploration
_SOFT_BESIEGE_THRESHOLD = 0.5  # |E| >= 0.5 triggers soft besiege
_RANDOM_THRESHOLD = 0.5  # Threshold for random decisions


class HarrisHawksOptimizer(AbstractOptimizer):
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

        >>> from opt.swarm_intelligence.harris_hawks_optimization import HarrisHawksOptimizer
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = HarrisHawksOptimizer(
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
        >>> optimizer = HarrisHawksOptimizer(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature:

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

    def _levy_flight(self, rng: np.random.Generator, dim: int) -> np.ndarray:
        """Generate Levy flight step using Mantegna's algorithm.

        Args:
            rng: NumPy random generator.
            dim: Dimensionality of the step.

        Returns:
            Levy flight step vector.
        """
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = rng.normal(0, sigma, dim)
        v = rng.normal(0, 1, dim)
        return u / (np.abs(v) ** (1 / beta))

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Harris Hawks Optimization algorithm.

        Returns:
            Tuple containing:
                - best_solution: The best solution found (numpy array).
                - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize hawk population
        hawks = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(hawk) for hawk in hawks])

        # Find initial prey (best solution)
        best_idx = np.argmin(fitness)
        prey = hawks[best_idx].copy()
        prey_fitness = fitness[best_idx]

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Update escaping energy E (decreases from 2 to 0)
            e0 = 2 * rng.random() - 1  # Initial energy in [-1, 1]
            escaping_energy = 2 * e0 * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                q = rng.random()
                r = rng.random()

                if abs(escaping_energy) >= _EXPLORATION_THRESHOLD:
                    # Exploration phase
                    if q >= _RANDOM_THRESHOLD:
                        # Perch based on random tall tree (random hawk)
                        rand_idx = rng.integers(0, self.population_size)
                        hawks[i] = hawks[rand_idx] - rng.random() * abs(
                            hawks[rand_idx] - 2 * rng.random() * hawks[i]
                        )
                    else:
                        # Perch on random tall tree on the edge of home territory
                        hawks[i] = (prey - hawks.mean(axis=0)) - rng.random() * (
                            self.lower_bound
                            + rng.random() * (self.upper_bound - self.lower_bound)
                        )
                # Exploitation phase - different strategies based on |E| and r
                elif r >= _RANDOM_THRESHOLD:
                    # Soft besiege (prey has energy to escape)
                    if abs(escaping_energy) >= _SOFT_BESIEGE_THRESHOLD:
                        # Soft besiege
                        jump_strength = 2 * (1 - rng.random())
                        hawks[i] = prey - escaping_energy * abs(
                            jump_strength * prey - hawks[i]
                        )
                    else:
                        # Hard besiege
                        jump_strength = 2 * (1 - rng.random())
                        hawks[i] = prey - escaping_energy * abs(prey - hawks[i])
                # Progressive rapid dives with Levy flight
                elif abs(escaping_energy) >= _SOFT_BESIEGE_THRESHOLD:
                    # Soft besiege with progressive rapid dives
                    jump_strength = 2 * (1 - rng.random())
                    y = prey - escaping_energy * abs(jump_strength * prey - hawks[i])
                    y = np.clip(y, self.lower_bound, self.upper_bound)

                    if self.func(y) < fitness[i]:
                        hawks[i] = y
                    else:
                        # Levy flight
                        z = y + rng.random(self.dim) * self._levy_flight(rng, self.dim)
                        z = np.clip(z, self.lower_bound, self.upper_bound)
                        if self.func(z) < fitness[i]:
                            hawks[i] = z
                else:
                    # Hard besiege with progressive rapid dives
                    jump_strength = 2 * (1 - rng.random())
                    y = prey - escaping_energy * abs(
                        jump_strength * prey - hawks.mean(axis=0)
                    )
                    y = np.clip(y, self.lower_bound, self.upper_bound)

                    if self.func(y) < fitness[i]:
                        hawks[i] = y
                    else:
                        # Levy flight
                        z = y + rng.random(self.dim) * self._levy_flight(rng, self.dim)
                        z = np.clip(z, self.lower_bound, self.upper_bound)
                        if self.func(z) < fitness[i]:
                            hawks[i] = z

                # Ensure bounds
                hawks[i] = np.clip(hawks[i], self.lower_bound, self.upper_bound)

                # Update fitness
                fitness[i] = self.func(hawks[i])

                # Update prey (best solution)
                if fitness[i] < prey_fitness:
                    prey = hawks[i].copy()
                    prey_fitness = fitness[i]

        return prey, prey_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(HarrisHawksOptimizer)
