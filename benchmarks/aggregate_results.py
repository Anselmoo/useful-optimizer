"""Aggregate benchmark results and generate summary statistics.

This script processes raw benchmark results from run_benchmark_suite.py and
generates aggregated statistics compatible with COCO/BBOB standards.

Output format follows the benchmark-data-schema.json schema and includes:
- Statistical summaries (mean, std, min, max, median)
- Expected Running Time (ERT) calculation
- Success rates at target precision
- Downsampled convergence curves for visualization
"""

from __future__ import annotations

import argparse
import json
import sys

from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np

from pydantic import ValidationError

from opt.benchmark.optima import get_optimum_safe


def downsample_convergence(
    history: list[float], target_points: int = 100
) -> list[float]:
    """Downsample convergence history to target number of points.

    Uses linear interpolation to reduce data size while preserving curve shape.

    Args:
        history: Full convergence history.
        target_points: Target number of points. Defaults to 100.

    Returns:
        Downsampled convergence history.

    Example:
        >>> history = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
        >>> downsampled = downsample_convergence(history, target_points=3)
        >>> len(downsampled)
        3
    """
    if len(history) <= target_points:
        return history

    indices = np.linspace(0, len(history) - 1, target_points, dtype=int)
    return [history[i] for i in indices]


def calculate_ert(
    runs: list[dict], target_precision: float = 1e-8, f_opt: float = 0.0
) -> float | None:
    """Calculate Expected Running Time (ERT).

    ERT is defined as the average number of function evaluations needed
    to reach target precision, accounting for runs that didn't converge.

    Formula: ERT = (total_evaluations) / (successful_runs / total_runs)

    Args:
        runs: List of run results.
        target_precision: Target precision threshold. Defaults to 1e-8.
        f_opt: Known optimal value. Defaults to 0.0.

    Returns:
        ERT value, or None if no successful runs.

    Example:
        >>> runs = [
        ...     {"best_fitness": 1e-9, "n_evaluations": 100},
        ...     {"best_fitness": 1e-7, "n_evaluations": 150},
        ... ]
        >>> ert = calculate_ert(runs, target_precision=1e-8, f_opt=0.0)
        >>> ert == 100.0
        True
    """
    if not runs:
        return None

    successful_runs = []
    for run in runs:
        best_fitness = run.get("best_fitness")
        n_evaluations = run.get("n_evaluations")

        if (
            best_fitness is not None
            and n_evaluations is not None
            and abs(best_fitness - f_opt) < target_precision
        ):
            successful_runs.append(n_evaluations)

    if not successful_runs:
        return None

    # ERT = average evaluations for successful runs / success rate
    avg_evals = np.mean(successful_runs)
    success_rate = len(successful_runs) / len(runs)

    return avg_evals / success_rate if success_rate > 0 else None


def compute_statistics(runs: list[dict]) -> dict:
    """Compute statistical summary across multiple runs.

    Args:
        runs: List of run results.

    Returns:
        Dictionary with statistical metrics.

    Example:
        >>> runs = [
        ...     {"best_fitness": 1.0, "n_evaluations": 100},
        ...     {"best_fitness": 2.0, "n_evaluations": 150},
        ...     {"best_fitness": 1.5, "n_evaluations": 125},
        ... ]
        >>> stats = compute_statistics(runs)
        >>> stats["mean_fitness"]
        1.5
    """
    if not runs:
        return {}

    fitness_values = [r["best_fitness"] for r in runs if "best_fitness" in r]

    if not fitness_values:
        return {}

    return {
        "mean_fitness": float(np.mean(fitness_values)),
        "std_fitness": float(np.std(fitness_values)),
        "min_fitness": float(np.min(fitness_values)),
        "max_fitness": float(np.max(fitness_values)),
        "median_fitness": float(np.median(fitness_values)),
        "q1_fitness": float(np.percentile(fitness_values, 25)),
        "q3_fitness": float(np.percentile(fitness_values, 75)),
    }


def aggregate_results(input_file: Path, _output_dir: Path | None = None) -> dict:
    """Aggregate raw benchmark results into summary statistics.

    Args:
        input_file: Path to raw results JSON file.
        output_dir: Directory to save processed results.

    Returns:
        Aggregated results dictionary.
    """
    with input_file.open() as f:
        raw_data = json.load(f)

    # Initialize processed results
    processed = {
        "metadata": {
            "max_iterations": raw_data["metadata"].get("max_iterations", 0),
            "n_runs": raw_data["metadata"].get("n_runs", 0),
            "dimensions": raw_data["metadata"].get("dimensions", []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
        },
        "benchmarks": {},
    }

    # Process each function/dimension/optimizer combination
    for func_name, func_data in raw_data.get("benchmarks", {}).items():
        processed["benchmarks"][func_name] = {}
        f_opt = get_optimum_safe(func_name, default=0.0)

        for dim_key, dim_data in func_data.items():
            processed["benchmarks"][func_name][dim_key] = {}

            for optimizer_name, optimizer_data in dim_data.items():
                runs = optimizer_data.get("runs", [])

                if not runs:
                    continue

                # Compute statistics
                stats = compute_statistics(runs)

                # Calculate ERT
                ert = calculate_ert(
                    runs,
                    target_precision=1e-8,
                    f_opt=f_opt if f_opt is not None else 0.0,
                )

                # Calculate success rate
                successful = sum(
                    1
                    for r in runs
                    if "best_fitness" in r
                    and abs(r["best_fitness"] - (f_opt or 0.0)) < 1e-8
                )
                success_rate = successful / len(runs) if runs else 0.0

                # Process convergence histories
                processed_runs = []
                for run in runs:
                    processed_run = {
                        "best_fitness": run.get("best_fitness", float("inf")),
                        "best_solution": run.get("best_solution", []),
                        "n_evaluations": run.get("n_evaluations", 0),
                    }

                    # Downsample convergence history if available
                    if run.get("convergence_history"):
                        downsampled = downsample_convergence(
                            run["convergence_history"], target_points=100
                        )
                        processed_run["history"] = {"best_fitness": downsampled}

                    processed_runs.append(processed_run)

                # Store processed data
                processed["benchmarks"][func_name][dim_key][optimizer_name] = {
                    "runs": processed_runs,
                    "statistics": stats,
                    "success_rate": success_rate,
                }

                # Add ERT to statistics if available
                if ert is not None and stats:
                    stats["ert"] = ert

    return processed


def validate_schema(data: dict) -> bool:
    """Validate aggregated data against Pydantic schema.

    Args:
        data: Aggregated results dictionary.

    Returns:
        True if valid, False otherwise.
    """
    try:
        from benchmarks.models import BenchmarkDataSchema

        BenchmarkDataSchema(**data)
    except ValidationError as e:
        print(f"Schema validation failed: {e}", file=sys.stderr)
        return False
    except ImportError:
        print(
            "Warning: Could not import BenchmarkDataSchema for validation",
            file=sys.stderr,
        )
    return True  # Success or skipped validation


def main():
    """Main entry point for aggregation script."""
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results into summary statistics"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmarks/output/results.json"),
        help="Input raw results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/output/processed"),
        help="Output directory for processed results",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate output against schema"
    )

    args = parser.parse_args()

    # Check input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate results
    print(f"Processing {args.input}...")
    processed_data = aggregate_results(args.input, args.output_dir)

    # Validate if requested
    if args.validate:
        print("Validating against schema...")
        if not validate_schema(processed_data):
            print("Validation failed!", file=sys.stderr)
            sys.exit(1)
        print("Validation passed!")

    # Save processed results
    output_file = args.output_dir / "benchmark-summary.json"
    with output_file.open("w") as f:
        json.dump(processed_data, f, indent=2)

    print(f"Processed results saved to {output_file}")

    # Print summary
    n_functions = len(processed_data.get("benchmarks", {}))
    n_optimizers = sum(
        len(dim_data)
        for func_data in processed_data.get("benchmarks", {}).values()
        for dim_data in func_data.values()
    )

    print("\nSummary:")
    print(f"  Functions: {n_functions}")
    print(f"  Optimizer-dimension combinations: {n_optimizers}")
    print(f"  Runs per combination: {processed_data['metadata']['n_runs']}")


if __name__ == "__main__":
    main()
