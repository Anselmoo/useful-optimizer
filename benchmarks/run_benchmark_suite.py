r"""Benchmark suite runner for optimization algorithms.

This script runs optimization algorithms on standard benchmark functions with
tiered optimizer selection for different use cases:

- SHOWCASE (4 optimizers): Fast PR validation, ~15-30 min
- STANDARD (13 optimizers): Comprehensive testing, ~6-8 hours
- COMPREHENSIVE (120+ optimizers): Deep analysis, ~55-70 hours [future work]

Why not all 120+ optimizers by default?
- 43,200+ benchmark runs (120 $\times$ 6 functions $\times$ 4 dims $\times$ 15 runs)
- 55-70 hour runtime makes CI impractical
- 500MB-1GB artifact size exceeds GitHub limits
- The 13-optimizer standard tier provides representative coverage
  across all algorithm families (classical, evolutionary, gradient-based,
  metaheuristic, swarm intelligence)

Usage:
    # Quick PR check
    python benchmarks/run_benchmark_suite.py --tier showcase

    # Daily comprehensive testing
    python benchmarks/run_benchmark_suite.py --tier standard
"""

from __future__ import annotations

import json
import time

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pydantic import BaseModel
from pydantic import Field

from opt.benchmark.functions import ackley
from opt.benchmark.functions import griewank
from opt.benchmark.functions import rastrigin
from opt.benchmark.functions import rosenbrock
from opt.benchmark.functions import shifted_ackley
from opt.benchmark.functions import sphere

# Optimizers
from opt.classical.hill_climbing import HillClimbing
from opt.classical.nelder_mead import NelderMead
from opt.classical.simulated_annealing import SimulatedAnnealing
from opt.evolutionary.differential_evolution import DifferentialEvolution
from opt.evolutionary.genetic_algorithm import GeneticAlgorithm
from opt.gradient_based.adamw import AdamW
from opt.gradient_based.sgd_momentum import SGDMomentum
from opt.metaheuristic.harmony_search import HarmonySearch
from opt.swarm_intelligence.ant_colony import AntColony
from opt.swarm_intelligence.bat_algorithm import BatAlgorithm
from opt.swarm_intelligence.firefly_algorithm import FireflyAlgorithm
from opt.swarm_intelligence.grey_wolf_optimizer import GreyWolfOptimizer
from opt.swarm_intelligence.particle_swarm import ParticleSwarm


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


# Pydantic models for structured results
class BenchmarkMetadata(BaseModel):
    """Metadata for benchmark execution."""

    max_iterations: int = Field(ge=1)
    n_runs: int = Field(ge=1)
    dimensions: list[int]
    timestamp: str
    target_precision: float = Field(gt=0)
    subset: bool
    python_version: str | None = None
    numpy_version: str | None = None


class RunResult(BaseModel):
    """Individual run result."""

    optimizer: str
    best_fitness: float
    best_solution: list[float]
    elapsed_time: float
    n_evaluations: int
    converged: bool
    evaluations_to_target: int | None = None
    convergence_history: list[float] | None = None
    status: str
    error: str | None = None


class OptimizerStatistics(BaseModel):
    """Statistics for an optimizer on a function."""

    mean_fitness: float
    std_fitness: float
    min_fitness: float
    max_fitness: float
    median_fitness: float
    mean_time: float
    std_time: float
    mean_evaluations: float
    std_evaluations: float


class OptimizerResults(BaseModel):
    """Results for an optimizer on a specific function and dimension."""

    runs: list[RunResult]
    statistics: OptimizerStatistics | None = None
    success_rate: float = Field(ge=0.0, le=1.0)


class BenchmarkResults(BaseModel):
    """Complete benchmark results."""

    metadata: BenchmarkMetadata
    benchmarks: dict[str, dict[str, dict[str, OptimizerResults]]]


# Configuration
BENCHMARK_FUNCTIONS = {
    "sphere": {"func": sphere, "bounds": (-5.12, 5.12), "f_opt": 0.0},
    "rosenbrock": {"func": rosenbrock, "bounds": (-5.0, 10.0), "f_opt": 0.0},
    "rastrigin": {"func": rastrigin, "bounds": (-5.12, 5.12), "f_opt": 0.0},
    "ackley": {"func": ackley, "bounds": (-32.768, 32.768), "f_opt": 0.0},
    "shifted_ackley": {
        "func": shifted_ackley,
        "bounds": (-32.768, 32.768),
        "f_opt": 0.0,
    },
    "griewank": {"func": griewank, "bounds": (-600.0, 600.0), "f_opt": 0.0},
}

# Tiered optimizer configuration for different benchmark scenarios
# Tiers balance computation time vs comprehensive coverage
OPTIMIZER_TIERS = {
    # SHOWCASE: 4 algorithms, ~15-30 min runtime
    # Purpose: Fast PR validation, CI checks
    # Coverage: One representative from each major category
    "showcase": {
        "ParticleSwarm": ParticleSwarm,  # Swarm Intelligence
        "DifferentialEvolution": DifferentialEvolution,  # Evolutionary
        "AdamW": AdamW,  # Gradient-based
        "HarmonySearch": HarmonySearch,  # Metaheuristic
    },
    # STANDARD: 13 algorithms, ~6-8 hours runtime
    # Purpose: Daily validation, comprehensive testing
    # Coverage: Representative sample across all algorithm families
    "standard": {
        "ParticleSwarm": ParticleSwarm,
        "AntColony": AntColony,
        "FireflyAlgorithm": FireflyAlgorithm,
        "BatAlgorithm": BatAlgorithm,
        "GreyWolfOptimizer": GreyWolfOptimizer,
        "GeneticAlgorithm": GeneticAlgorithm,
        "DifferentialEvolution": DifferentialEvolution,
        "HarmonySearch": HarmonySearch,
        "SimulatedAnnealing": SimulatedAnnealing,
        "HillClimbing": HillClimbing,
        "NelderMead": NelderMead,
        "AdamW": AdamW,
        "SGDMomentum": SGDMomentum,
    },
    # COMPREHENSIVE: All 120+ algorithms, ~55-70 hours runtime
    # Purpose: Monthly/quarterly deep analysis
    # Coverage: Complete algorithm library
    # Note: Requires dynamic import of all optimizer classes
    "comprehensive": "all",  # Placeholder - requires dynamic loading
}

# Runtime estimates (6 functions $\times$ [2,5,10,20] dims $\times$ 15 runs)
TIER_RUNTIMES = {
    "showcase": "15-30 minutes (4 optimizers, 1,440 runs)",
    "standard": "6-8 hours (13 optimizers, 4,680 runs)",
    "comprehensive": "55-70 hours (120+ optimizers, 43,200+ runs)",
}

# Legacy aliases for backward compatibility
OPTIMIZERS = OPTIMIZER_TIERS["standard"]  # Default to standard tier
SHOWCASE_OPTIMIZERS = OPTIMIZER_TIERS["showcase"]  # Deprecated

DIMENSIONS = [2, 5, 10, 20]  # BBOB standard dimensions
MAX_ITERATIONS = 1000  # Increased for better convergence
N_RUNS = 15  # BBOB standard: 15 independent runs
TARGET_PRECISION = 1e-8  # BBOB standard target precision


def run_single_benchmark(
    optimizer_class: type,
    func: Callable[[ndarray], float],
    lower_bound: float,
    upper_bound: float,
    dim: int,
    max_iter: int = MAX_ITERATIONS,
    seed: int | None = None,
    f_opt: float = 0.0,
    target_precision: float = TARGET_PRECISION,
) -> dict:
    """Run a single benchmark test.

    Args:
        optimizer_class: Optimizer class to instantiate
        func: Benchmark function
        lower_bound: Lower bound of search space
        upper_bound: Upper bound of search space
        dim: Dimensionality of the problem
        max_iter: Maximum iterations
        seed: Random seed for reproducibility
        f_opt: Known optimal value for convergence checking
        target_precision: Target precision for early stopping

    Returns:
        dict: Results containing solution, fitness, evaluations, and timing information
    """
    if seed is not None:
        np.random.seed(seed)

    # Create optimizer with history tracking and COCO/BBOB parameters
    try:
        # Determine if optimizer needs population_size
        import inspect

        sig = inspect.signature(optimizer_class.__init__)
        needs_population = "population_size" in sig.parameters

        kwargs = {
            "func": func,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "dim": dim,
            "max_iter": max_iter,
            "track_history": True,
            "target_precision": target_precision,
            "f_opt": f_opt,
            "seed": seed,
        }

        if needs_population:
            kwargs["population_size"] = 30  # Moderate population size

        optimizer = optimizer_class(**kwargs)
    except TypeError as e:
        return {
            "error": str(e),
            "optimizer": optimizer_class.__name__,
            "status": "failed",
        }

    # Run optimization
    start_time = time.time()
    try:
        best_solution, best_fitness = optimizer.search()
        elapsed_time = time.time() - start_time

        # Extract convergence history if available
        convergence_history = None
        if optimizer.track_history and optimizer.history.get("best_fitness"):
            convergence_history = optimizer.history["best_fitness"]

        return {
            "optimizer": optimizer_class.__name__,
            "best_fitness": float(best_fitness),
            "best_solution": best_solution.tolist(),
            "elapsed_time": elapsed_time,
            "n_evaluations": optimizer.n_evaluations,
            "converged": optimizer.converged,
            "evaluations_to_target": optimizer.evaluations_to_target,
            "convergence_history": convergence_history,
            "status": "success",
        }
    except Exception as e:
        return {
            "error": str(e),
            "optimizer": optimizer_class.__name__,
            "status": "failed",
        }


def run_benchmark_suite(
    output_dir: str | Path = "benchmarks/output",
    tier: str = "standard",
    subset: bool | None = None,  # Deprecated - use tier instead  # noqa: FBT001
) -> dict:
    """Run complete benchmark suite with tiered optimizer selection.

    Args:
        output_dir: Directory to save results
        tier: Optimizer tier - 'showcase' (4 algos, ~30min), 'standard' (13 algos, ~6h),
              or 'comprehensive' (120+ algos, ~60h). See TIER_RUNTIMES for details.
        subset: Deprecated - use tier='showcase' instead. Kept for backward compatibility.

    Returns:
        dict: Complete benchmark results with metadata about tier used

    Raises:
        ValueError: If tier is invalid or 'comprehensive' tier not yet implemented
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle backward compatibility with deprecated --subset flag
    if subset is not None:
        tier = "showcase" if subset else "standard"
        print(
            "Warning: --subset flag is deprecated. Use --tier showcase/standard/comprehensive instead."
        )

    # Validate and select optimizer set
    if tier not in OPTIMIZER_TIERS:
        msg = (
            f"Invalid tier '{tier}'. "
            f"Must be one of: {', '.join(OPTIMIZER_TIERS.keys())}"
        )
        raise ValueError(msg)

    if tier == "comprehensive":
        msg = (
            "Comprehensive tier (120+ optimizers) not yet implemented.\n"
            "This requires dynamic loading of all optimizer classes.\n"
            "Use 'standard' tier for now (13 representative optimizers)."
        )
        raise NotImplementedError(msg)

    optimizers = OPTIMIZER_TIERS[tier]
    print(f"\nRunning {tier.upper()} tier: {len(optimizers)} optimizers")
    print(f"Estimated runtime: {TIER_RUNTIMES[tier]}\n")

    results = {
        "metadata": {
            "max_iterations": MAX_ITERATIONS,
            "n_runs": N_RUNS,
            "dimensions": DIMENSIONS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_precision": TARGET_PRECISION,
            "tier": tier,
            "n_optimizers": len(optimizers),
            "estimated_runtime": TIER_RUNTIMES[tier],
        },
        "benchmarks": {},
    }

    benchmarks_dict: dict[str, dict[str, dict[str, OptimizerResults]]] = {}

    total_tests = len(BENCHMARK_FUNCTIONS) * len(optimizers) * len(DIMENSIONS) * N_RUNS
    test_count = 0

    for func_name, func_config in BENCHMARK_FUNCTIONS.items():
        benchmarks_dict[func_name] = {}

        for dim in DIMENSIONS:
            benchmarks_dict[func_name][f"{dim}D"] = {}

            for optimizer_name, optimizer_class in optimizers.items():
                print(
                    f"Running {optimizer_name} on {func_name} ({dim}D) "
                    f"[{test_count}/{total_tests}]..."
                )

                run_results = []
                for run_idx in range(N_RUNS):
                    result_dict = run_single_benchmark(
                        optimizer_class=optimizer_class,
                        func=func_config["func"],
                        lower_bound=func_config["bounds"][0],
                        upper_bound=func_config["bounds"][1],
                        dim=dim,
                        max_iter=MAX_ITERATIONS,
                        seed=run_idx,  # Use run index as seed for reproducibility
                        f_opt=func_config.get("f_opt", 0.0),
                        target_precision=TARGET_PRECISION,
                    )
                    # Convert dict to Pydantic model
                    run_result = RunResult(**result_dict)
                    run_results.append(run_result)
                    test_count += 1

                # Compute statistics across runs
                successful_runs = [r for r in run_results if r.status == "success"]

                if successful_runs:
                    fitness_values = [r.best_fitness for r in successful_runs]
                    time_values = [r.elapsed_time for r in successful_runs]
                    eval_values = [r.n_evaluations for r in successful_runs]

                    statistics = OptimizerStatistics(
                        mean_fitness=float(np.mean(fitness_values)),
                        std_fitness=float(np.std(fitness_values)),
                        min_fitness=float(np.min(fitness_values)),
                        max_fitness=float(np.max(fitness_values)),
                        median_fitness=float(np.median(fitness_values)),
                        mean_time=float(np.mean(time_values)),
                        std_time=float(np.std(time_values)),
                        mean_evaluations=float(np.mean(eval_values)),
                        std_evaluations=float(np.std(eval_values)),
                    )
                    success_rate = len(successful_runs) / N_RUNS
                else:
                    statistics = None
                    success_rate = 0.0

                optimizer_results = OptimizerResults(
                    runs=run_results, statistics=statistics, success_rate=success_rate
                )

                benchmarks_dict[func_name][f"{dim}D"][optimizer_name] = (
                    optimizer_results
                )

    # Create final Pydantic model
    metadata = BenchmarkMetadata(
        max_iterations=MAX_ITERATIONS,
        n_runs=N_RUNS,
        dimensions=DIMENSIONS,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        target_precision=TARGET_PRECISION,
        tier=tier,
        n_optimizers=len(optimizers),
        estimated_runtime=TIER_RUNTIMES[tier],
    )
    results = BenchmarkResults(metadata=metadata, benchmarks=benchmarks_dict)

    # Save results to JSON
    output_file = output_dir / "results.json"
    with output_file.open("w") as f:
        json.dump(results.model_dump(), f, indent=2)

    print(f"\nBenchmark suite completed. Results saved to {output_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run optimization benchmark suite with tiered optimizer selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tier Details:
  showcase      : 4 optimizers, ~15-30 min   (PR checks, quick validation)
  standard      : 13 optimizers, ~6-8 hours  (daily/weekly comprehensive testing)
  comprehensive : 120+ optimizers, ~55-70h   (monthly deep analysis) [NOT YET IMPLEMENTED]

Examples:
  # Quick PR validation
  python benchmarks/run_benchmark_suite.py --tier showcase

  # Daily comprehensive testing
  python benchmarks/run_benchmark_suite.py --tier standard

  # Custom output directory
  python benchmarks/run_benchmark_suite.py --tier showcase --output-dir /tmp/bench
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/output",
        help="Output directory for results (default: benchmarks/output)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["showcase", "standard", "comprehensive"],
        default="standard",
        help="Optimizer tier selection (default: standard)",
    )
    # Deprecated - kept for backward compatibility
    parser.add_argument(
        "--subset", action="store_true", help="[DEPRECATED] Use --tier showcase instead"
    )
    args = parser.parse_args()

    results = run_benchmark_suite(
        output_dir=args.output_dir,
        tier=args.tier,
        subset=args.subset if args.subset else None,
    )

    print(
        f"\nCompleted {len(results['benchmarks'])} functions x "
        f"{results['metadata']['n_optimizers']} optimizers x "
        f"{len(DIMENSIONS)} dimensions x {N_RUNS} runs"
    )
    print(f"Tier: {results['metadata']['tier'].upper()}")
    print(f"Estimated runtime: {results['metadata']['estimated_runtime']}")
