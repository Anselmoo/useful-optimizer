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
from opt.swarm_intelligence.ant_colony import AntColony
from opt.swarm_intelligence.bat_algorithm import BatAlgorithm
from opt.swarm_intelligence.firefly_algorithm import FireflyAlgorithm
from opt.swarm_intelligence.grey_wolf_optimizer import GreyWolfOptimizer
from opt.swarm_intelligence.particle_swarm import ParticleSwarm

# Classical
from opt.classical.hill_climbing import HillClimbing
from opt.classical.nelder_mead import NelderMead
from opt.classical.simulated_annealing import SimulatedAnnealing

# Evolutionary
from opt.evolutionary.differential_evolution import DifferentialEvolution
from opt.evolutionary.genetic_algorithm import GeneticAlgorithm

# Gradient-based (for comparison)
from opt.gradient_based.adamw import AdamW
from opt.gradient_based.sgd_momentum import SGDMomentum

# Metaheuristic
from opt.metaheuristic.harmony_search import HarmonySearch


# Configuration
BENCHMARK_FUNCTIONS = {
    "sphere": {"func": sphere, "bounds": (-5.12, 5.12)},
    "rosenbrock": {"func": rosenbrock, "bounds": (-5.0, 10.0)},
    "rastrigin": {"func": rastrigin, "bounds": (-5.12, 5.12)},
    "ackley": {"func": ackley, "bounds": (-32.768, 32.768)},
    "shifted_ackley": {"func": shifted_ackley, "bounds": (-32.768, 32.768)},
    "griewank": {"func": griewank, "bounds": (-600.0, 600.0)},
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

DIMENSIONS = [2, 10, 30]
MAX_ITERATIONS = 100
N_RUNS = 10  # Number of runs per configuration for stability


def run_single_benchmark(
    optimizer_class: type,
    func: Callable[[ndarray], float],
    lower_bound: float,
    upper_bound: float,
    dim: int,
    max_iter: int = MAX_ITERATIONS,
    seed: int | None = None,
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

    Returns:
        dict: Results containing solution, fitness, and timing information
    """
    if seed is not None:
        np.random.seed(seed)

    # Create optimizer with history tracking for convergence analysis
    try:
        optimizer = optimizer_class(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            track_history=True,
            population_size=30,
        )
    except TypeError:
        # Some optimizers don't support track_history or population_size
        try:
            optimizer = optimizer_class(
                func=func,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                dim=dim,
                max_iter=max_iter,
            )
        except Exception as e:
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
        if hasattr(optimizer, "best_fitness_history"):
            convergence_history = optimizer.best_fitness_history

        return {
            "optimizer": optimizer_class.__name__,
            "best_fitness": float(best_fitness),
            "best_solution": best_solution.tolist(),
            "elapsed_time": elapsed_time,
            "convergence_history": convergence_history,
            "status": "success",
        }
    except Exception as e:
        return {
            "error": str(e),
            "optimizer": optimizer_class.__name__,
            "status": "failed",
        }


def run_benchmark_suite(output_dir: str | Path = "benchmarks/output") -> dict:
    """Run complete benchmark suite.

    Args:
        output_dir: Directory to save results

    Returns:
        dict: Complete benchmark results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "max_iterations": MAX_ITERATIONS,
            "n_runs": N_RUNS,
            "dimensions": DIMENSIONS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "benchmarks": {},
    }

    total_tests = len(BENCHMARK_FUNCTIONS) * len(OPTIMIZERS) * len(DIMENSIONS) * N_RUNS
    test_count = 0

    for func_name, func_config in BENCHMARK_FUNCTIONS.items():
        results["benchmarks"][func_name] = {}

        for dim in DIMENSIONS:
            results["benchmarks"][func_name][f"{dim}D"] = {}

            for optimizer_name, optimizer_class in OPTIMIZERS.items():
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
                        seed=42 + run_idx,  # Different seed per run
                    )
                    run_results.append(result)
                    test_count += 1

                # Compute statistics across runs
                successful_runs = [r for r in run_results if r["status"] == "success"]

                if successful_runs:
                    fitness_values = [r["best_fitness"] for r in successful_runs]
                    time_values = [r["elapsed_time"] for r in successful_runs]

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
    args = parser.parse_args()

    results = run_benchmark_suite(output_dir=args.output_dir)
    print(
        f"\nCompleted {len(results['benchmarks'])} functions x "
        f"{len(OPTIMIZERS)} optimizers x {len(DIMENSIONS)} dimensions x {N_RUNS} runs"
    )
