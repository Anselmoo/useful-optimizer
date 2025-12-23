"""Dragonfly Algorithm (DA).

This module implements the Dragonfly Algorithm, a swarm intelligence optimization
algorithm based on the static and dynamic swarming behaviors of dragonflies.

Dragonflies form sub-swarms for hunting (static swarm) and migrate in one direction
(dynamic swarm). These behaviors map to exploration and exploitation in optimization.

Reference:
    Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization
    technique for solving single-objective, discrete, and multi-objective problems.
    Neural Computing and Applications, 27(4), 1053-1073.
    DOI: 10.1007/s00521-015-1920-1

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = DragonflyOptimizer(
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
    population_size (int): Number of dragonflies in the swarm.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class DragonflyOptimizer(AbstractOptimizer):
    r"""Dragonfly Algorithm (DA) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | Dragonfly Algorithm                      |
        | Acronym           | DA                                       |
        | Year Introduced   | 2016                                     |
        | Authors           | Mirjalili, Seyedali                      |
        | Algorithm Class   | Swarm Intelligence                       |
        | Complexity        | O(population_size * dim * max_iter)      |
        | Properties        | Population-based, Static/Dynamic swarming, Derivative-free |
        | Implementation    | Python 3.10+                             |
        | COCO Compatible   | Yes                                      |

    Mathematical Formulation:
        Core update equations based on dragonfly swarming behavior:

        Step velocity:
            $$
            \Delta X_{t+1} = (sS_i + aA_i + cC_i + fF_i + eE_i) + w\Delta X_t
            $$

        Position update:
            $$
            X_{t+1} = X_t + \Delta X_{t+1}
            $$

        where:
            - $S_i$ is separation (avoid crowding)
            - $A_i$ is alignment (velocity matching)
            - $C_i$ is cohesion (tendency to center)
            - $F_i$ is food factor (attraction to prey/best solution)
            - $E_i$ is enemy factor (distraction from worst)
            - $s, a, c, f, e$ are weights for each component
            - $w$ is inertia weight
            - Weights adapt over iterations to balance exploration/exploitation

        Constraint handling:
            - **Boundary conditions**: Clamping to [lower_bound, upper_bound]
            - **Feasibility enforcement**: Position updates maintain bounds

        Constraint handling:
            - **Boundary conditions**: FIXME: [clamping/reflection/periodic]
            - **Feasibility enforcement**: FIXME: [description]

    Hyperparameters:
        | Parameter              | Default | BBOB Recommended | Description                    |
        |------------------------|---------|------------------|--------------------------------|
        | population_size        | 30      | 10*dim           | Number of dragonflies          |
        | max_iter               | 1000    | 10000            | Maximum iterations             |

        **Sensitivity Analysis**:
            - Weights (s, a, c, f, e): **High** impact - control behavior components
            - Inertia w: **Medium** impact - balances exploration/exploitation
            - Recommended: Use adaptive weights (default behavior)

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

        >>> from opt.swarm_intelligence.dragonfly_algorithm import DragonflyOptimizer
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = DragonflyOptimizer(
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
        >>> optimizer = DragonflyOptimizer(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: func, lower_bound, upper_bound, dim, max_iter, seed, population_size

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

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 100,
    ) -> None:
        """Initialize the Dragonfly Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Problem dimensionality.
            max_iter: Maximum iterations.
            seed: Random seed.
            population_size: Number of dragonflies.
        """
        super().__init__(
            func, lower_bound, upper_bound, dim, max_iter, seed, population_size
        )

    def _find_neighbors(
        self, position: np.ndarray, all_positions: np.ndarray, radius: float
    ) -> np.ndarray:
        """Find neighbors within radius.

        Args:
            position: Current dragonfly position.
            all_positions: All dragonfly positions.
            radius: Neighborhood radius.

        Returns:
            Array of neighbor positions.
        """
        distances = np.linalg.norm(all_positions - position, axis=1)
        neighbor_mask = (distances < radius) & (distances > 0)
        return all_positions[neighbor_mask]

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Dragonfly Algorithm.

        Returns:
            Tuple containing:
        - best_solution: The best solution found (numpy array).
        - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize dragonfly population and velocities
        dragonflies = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        velocities = np.zeros((self.population_size, self.dim))

        # Evaluate initial fitness
        fitness = np.array([self.func(df) for df in dragonflies])

        # Track best (food) and worst (enemy) solutions
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)
        food = dragonflies[best_idx].copy()
        food_fitness = fitness[best_idx]
        enemy = dragonflies[worst_idx].copy()

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Update weights (decrease exploration, increase exploitation)
            w = 0.9 - iteration * ((0.9 - 0.4) / self.max_iter)
            # Update radius (decreases over iterations)
            radius = (self.upper_bound - self.lower_bound) * 0.1 + (
                (self.upper_bound - self.lower_bound)
                * (self.max_iter - iteration)
                / self.max_iter
            )

            # Adaptive parameters (increase over iterations)
            s = 2 * rng.random() * (iteration / self.max_iter)  # separation
            a = 2 * rng.random() * (iteration / self.max_iter)  # alignment
            c = 2 * rng.random() * (iteration / self.max_iter)  # cohesion
            f = 2 * rng.random()  # food attraction
            e = rng.random() * (1 - iteration / self.max_iter)  # enemy distraction

            for i in range(self.population_size):
                neighbors = self._find_neighbors(dragonflies[i], dragonflies, radius)

                if len(neighbors) > 0:
                    # Separation: avoid neighbors
                    separation = np.sum(neighbors - dragonflies[i], axis=0)

                    # Alignment: match velocity with neighbors
                    alignment = np.mean(velocities, axis=0)

                    # Cohesion: move toward neighbor center
                    cohesion = np.mean(neighbors, axis=0) - dragonflies[i]

                    # Food attraction
                    food_attraction = food - dragonflies[i]

                    # Enemy distraction
                    enemy_distraction = enemy + dragonflies[i]

                    # Update velocity
                    velocities[i] = (
                        w * velocities[i]
                        + s * separation
                        + a * alignment
                        + c * cohesion
                        + f * food_attraction
                        + e * enemy_distraction
                    )

                    # Update position
                    dragonflies[i] = dragonflies[i] + velocities[i]
                else:
                    # Levy flight for isolated dragonflies
                    levy = self._levy_flight(rng)
                    dragonflies[i] = dragonflies[i] + levy * dragonflies[i]

                # Ensure bounds
                dragonflies[i] = np.clip(
                    dragonflies[i], self.lower_bound, self.upper_bound
                )

            # Update fitness
            fitness = np.array([self.func(df) for df in dragonflies])

            # Update food (best) and enemy (worst)
            best_idx = np.argmin(fitness)
            worst_idx = np.argmax(fitness)

            if fitness[best_idx] < food_fitness:
                food = dragonflies[best_idx].copy()
                food_fitness = fitness[best_idx]

            enemy = dragonflies[worst_idx].copy()

        return food, food_fitness

    def _levy_flight(self, rng: np.random.Generator) -> np.ndarray:
        """Generate Levy flight step.

        Args:
            rng: Random number generator.

        Returns:
            Levy flight step vector.
        """
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = rng.normal(0, sigma, self.dim)
        v = rng.normal(0, 1, self.dim)

        return u / (np.abs(v) ** (1 / beta))


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(DragonflyOptimizer)
