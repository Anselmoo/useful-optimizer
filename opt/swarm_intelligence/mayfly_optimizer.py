"""Mayfly Optimization Algorithm.

Implementation based on:
Zervoudakis, K. & Tsafarakis, S. (2020).
A mayfly optimization algorithm.
Computers & Industrial Engineering, 145, 106559.

The algorithm mimics the mating behavior of mayflies, including nuptial
dances performed by males to attract females and the swarm dynamics of
both male and female mayflies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_A1 = 1.0  # Cognitive coefficient for males
_A2 = 1.5  # Social coefficient for males
_A3 = 1.5  # Attraction coefficient for females
_BETA = 2.0  # Attraction exponent
_DANCE = 5.0  # Nuptial dance coefficient
_FL = 0.1  # Random flight coefficient
_G = 0.8  # Gravity coefficient


class MayflyOptimizer(AbstractOptimizer):
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

        >>> from opt.swarm_intelligence.mayfly_optimizer import MayflyOptimizer
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = MayflyOptimizer(
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
        >>> optimizer = MayflyOptimizer(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: func, lower_bound, upper_bound, dim, max_iter, population_size, a1, a2, a3, beta, dance, fl, g

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
        max_iter: int,
        population_size: int = 30,
        a1: float = _A1,
        a2: float = _A2,
        a3: float = _A3,
        beta: float = _BETA,
        dance: float = _DANCE,
        fl: float = _FL,
        g: float = _G,
    ) -> None:
        """Initialize the MayflyOptimizer optimizer."""
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.n_males = population_size // 2
        self.n_females = population_size - self.n_males
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.beta = beta
        self.dance = dance
        self.fl = fl
        self.g = g

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Mayfly Optimization Algorithm.

        Returns:
        Tuple of (best_solution, best_fitness).
        """
        # Initialize male mayfly positions and velocities
        males = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.n_males, self.dim)
        )
        male_vel = np.zeros((self.n_males, self.dim))

        # Initialize female mayfly positions and velocities
        females = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.n_females, self.dim)
        )
        female_vel = np.zeros((self.n_females, self.dim))

        # Evaluate fitness
        male_fitness = np.array([self.func(m) for m in males])
        female_fitness = np.array([self.func(f) for f in females])

        # Personal bests for males
        male_pbest = males.copy()
        male_pbest_fitness = male_fitness.copy()

        # Global best
        all_positions = np.vstack([males, females])
        all_fitness = np.concatenate([male_fitness, female_fitness])
        best_idx = np.argmin(all_fitness)
        best_solution = all_positions[best_idx].copy()
        best_fitness = all_fitness[best_idx]

        for iteration in range(self.max_iter):
            # Update damping coefficient
            damp = 0.95 - 0.5 * (iteration / self.max_iter)

            # Update male positions
            for i in range(self.n_males):
                r1, r2 = np.random.rand(2)

                # Cognitive component
                cognitive = self.a1 * r1 * (male_pbest[i] - males[i])
                # Social component
                social = self.a2 * r2 * (best_solution - males[i])

                # Nuptial dance component
                if male_fitness[i] < best_fitness:
                    dance_component = self.dance * np.random.randn(self.dim)
                else:
                    dance_component = 0

                # Update velocity
                male_vel[i] = (
                    self.g * male_vel[i] + cognitive + social + dance_component
                )
                male_vel[i] *= damp

                # Update position
                males[i] = males[i] + male_vel[i]

                # Boundary handling
                males[i] = np.clip(males[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = self.func(males[i])
                male_fitness[i] = new_fitness

                # Update personal best
                if new_fitness < male_pbest_fitness[i]:
                    male_pbest[i] = males[i].copy()
                    male_pbest_fitness[i] = new_fitness

            # Update female positions
            for i in range(self.n_females):
                # Female moves toward corresponding male
                male_idx = i if i < self.n_males else i % self.n_males
                male_pos = males[male_idx]

                # Calculate distance
                distance = np.linalg.norm(females[i] - male_pos)

                if female_fitness[i] > male_fitness[male_idx]:
                    # Female is attracted to male
                    r3 = np.random.rand()
                    attraction = (
                        self.a3
                        * np.exp(-self.beta * distance**2)
                        * r3
                        * (male_pos - females[i])
                    )
                    female_vel[i] = self.g * female_vel[i] + attraction
                else:
                    # Random flight
                    female_vel[i] = self.g * female_vel[i] + self.fl * np.random.randn(
                        self.dim
                    )

                female_vel[i] *= damp

                # Update position
                females[i] = females[i] + female_vel[i]

                # Boundary handling
                females[i] = np.clip(females[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                female_fitness[i] = self.func(females[i])

            # Update global best
            for i in range(self.n_males):
                if male_fitness[i] < best_fitness:
                    best_solution = males[i].copy()
                    best_fitness = male_fitness[i]

            for i in range(self.n_females):
                if female_fitness[i] < best_fitness:
                    best_solution = females[i].copy()
                    best_fitness = female_fitness[i]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(MayflyOptimizer)
