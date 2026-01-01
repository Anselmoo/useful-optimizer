"""Benchmark suite runner for optimization algorithms.

This script runs all available optimization algorithms on standard benchmark
functions and outputs structured results for visualization.
"""

from __future__ import annotations

import json
import time

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from opt.benchmark.functions import ackley
from opt.benchmark.functions import griewank
from opt.benchmark.functions import rastrigin
from opt.benchmark.functions import rosenbrock
from opt.benchmark.functions import shifted_ackley
from opt.benchmark.functions import sphere


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


# Import optimizers from different categories
# Swarm Intelligence
# Classical
from opt.classical.hill_climbing import HillClimbing
from opt.classical.nelder_mead import NelderMead
from opt.classical.simulated_annealing import SimulatedAnnealing
from opt.evolutionary.differential_evolution import DifferentialEvolution

# Evolutionary
from opt.evolutionary.genetic_algorithm import GeneticAlgorithm

# Gradient-based (for comparison)
from opt.gradient_based.adamw import AdamW
from opt.gradient_based.sgd_momentum import SGDMomentum

# Metaheuristic
from opt.metaheuristic.harmony_search import HarmonySearch
from opt.swarm_intelligence.ant_colony import AntColony
from opt.swarm_intelligence.bat_algorithm import BatAlgorithm
from opt.swarm_intelligence.firefly_algorithm import FireflyAlgorithm
from opt.swarm_intelligence.grey_wolf_optimizer import GreyWolfOptimizer
from opt.swarm_intelligence.particle_swarm import ParticleSwarm


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

OPTIMIZERS = {
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
}

# Showcase subset for CI (fast, representative algorithms)
SHOWCASE_OPTIMIZERS = {
    "ParticleSwarm": ParticleSwarm,
    "DifferentialEvolution": DifferentialEvolution,
    "AdamW": AdamW,
    "HarmonySearch": HarmonySearch,
}

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
    output_dir: str | Path = "benchmarks/output", subset: bool = False
) -> dict:
    """Run complete benchmark suite.

    Args:
        output_dir: Directory to save results
        subset: If True, run only showcase optimizers for faster execution

    Returns:
        dict: Complete benchmark results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select optimizer set
    optimizers = SHOWCASE_OPTIMIZERS if subset else OPTIMIZERS

    results = {
        "metadata": {
            "max_iterations": MAX_ITERATIONS,
            "n_runs": N_RUNS,
            "dimensions": DIMENSIONS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_precision": TARGET_PRECISION,
            "subset": subset,
        },
        "benchmarks": {},
    }

    total_tests = len(BENCHMARK_FUNCTIONS) * len(optimizers) * len(DIMENSIONS) * N_RUNS
    test_count = 0

    for func_name, func_config in BENCHMARK_FUNCTIONS.items():
        results["benchmarks"][func_name] = {}

        for dim in DIMENSIONS:
            results["benchmarks"][func_name][f"{dim}D"] = {}

            for optimizer_name, optimizer_class in optimizers.items():
                print(
                    f"Running {optimizer_name} on {func_name} ({dim}D) "
                    f"[{test_count}/{total_tests}]..."
                )

                run_results = []
                for run_idx in range(N_RUNS):
                    result = run_single_benchmark(
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
                    run_results.append(result)
                    test_count += 1

                # Compute statistics across runs
                successful_runs = [r for r in run_results if r["status"] == "success"]

                if successful_runs:
                    fitness_values = [r["best_fitness"] for r in successful_runs]
                    time_values = [r["elapsed_time"] for r in successful_runs]
                    eval_values = [r["n_evaluations"] for r in successful_runs]

                    results["benchmarks"][func_name][f"{dim}D"][optimizer_name] = {
                        "runs": run_results,
                        "statistics": {
                            "mean_fitness": float(np.mean(fitness_values)),
                            "std_fitness": float(np.std(fitness_values)),
                            "min_fitness": float(np.min(fitness_values)),
                            "max_fitness": float(np.max(fitness_values)),
                            "median_fitness": float(np.median(fitness_values)),
                            "mean_time": float(np.mean(time_values)),
                            "std_time": float(np.std(time_values)),
                            "mean_evaluations": float(np.mean(eval_values)),
                            "std_evaluations": float(np.std(eval_values)),
                        },
                        "success_rate": len(successful_runs) / N_RUNS,
                    }
                else:
                    results["benchmarks"][func_name][f"{dim}D"][optimizer_name] = {
                        "runs": run_results,
                        "statistics": None,
                        "success_rate": 0.0,
                    }

    # Save results to JSON
    output_file = output_dir / "results.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark suite completed. Results saved to {output_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run optimization benchmark suite")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Run only showcase optimizers (PSO, DE, AdamW, HS) for faster execution",
    )
    args = parser.parse_args()

    results = run_benchmark_suite(output_dir=args.output_dir, subset=args.subset)

    optimizer_count = len(SHOWCASE_OPTIMIZERS) if args.subset else len(OPTIMIZERS)
    print(
        f"\nCompleted {len(results['benchmarks'])} functions x "
        f"{optimizer_count} optimizers x {len(DIMENSIONS)} dimensions x {N_RUNS} runs"
    )
