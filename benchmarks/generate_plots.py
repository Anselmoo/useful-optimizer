"""Generate visualization plots from benchmark results.

This script reads benchmark results and creates various visualizations
including convergence curves, performance heatmaps, and box plots.
"""

from __future__ import annotations

import json

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def load_results(results_path: str | Path) -> dict:
    """Load benchmark results from JSON file.

    Args:
        results_path: Path to results.json file

    Returns:
        dict: Benchmark results
    """
    with Path(results_path).open() as f:
        return json.load(f)


def plot_convergence_curves(
    results: dict,
    output_dir: str | Path,
    func_name: str = "shifted_ackley",
    dim: str = "2D",
) -> None:
    """Plot convergence curves for all optimizers on a specific function.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
        func_name: Function name to plot
        dim: Dimensionality (e.g., "2D", "10D")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _fig, ax = plt.subplots(figsize=(12, 8))

    if func_name not in results["benchmarks"]:
        print(f"Warning: Function {func_name} not found in results")
        return

    func_results = results["benchmarks"][func_name].get(dim, {})

    for optimizer_name, optimizer_data in func_results.items():
        if optimizer_data.get("runs"):
            # Get the first successful run with convergence history
            for run in optimizer_data["runs"]:
                if (
                    run["status"] == "success"
                    and run.get("convergence_history") is not None
                ):
                    history = run["convergence_history"]
                    ax.plot(history, label=optimizer_name, linewidth=2, alpha=0.8)
                    break

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    ax.set_title(f"Convergence Curves: {func_name} ({dim})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_yscale("log")
    ax.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"convergence_{func_name}_{dim}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved convergence plot to {output_file}")
    plt.close()


def plot_performance_heatmap(
    results: dict, output_dir: str | Path, dim: str = "2D", metric: str = "mean_fitness"
) -> None:
    """Plot performance heatmap showing optimizer performance across functions.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
        dim: Dimensionality to visualize
        metric: Metric to display (mean_fitness, min_fitness, mean_time)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data for heatmap
    functions = list(results["benchmarks"].keys())
    optimizers = set()

    # Collect all optimizer names
    for func_data in results["benchmarks"].values():
        if dim in func_data:
            optimizers.update(func_data[dim].keys())

    optimizers = sorted(optimizers)

    # Create matrix
    data = np.zeros((len(optimizers), len(functions)))
    data[:] = np.nan

    for j, func_name in enumerate(functions):
        func_data = results["benchmarks"][func_name].get(dim, {})
        for i, optimizer_name in enumerate(optimizers):
            if optimizer_name in func_data:
                stats = func_data[optimizer_name].get("statistics")
                if stats:
                    data[i, j] = stats.get(metric, np.nan)

    # Create heatmap
    _fig, ax = plt.subplots(figsize=(12, 8))

    # Use log scale for better visualization
    log_data = np.log10(data + 1e-10)  # Add small value to avoid log(0)

    sns.heatmap(
        log_data,
        annot=False,
        fmt=".2e",
        cmap="RdYlGn_r",
        xticklabels=functions,
        yticklabels=optimizers,
        cbar_kws={"label": f"log10({metric})"},
        ax=ax,
    )

    ax.set_title(f"Performance Heatmap ({dim}) - {metric}")
    ax.set_xlabel("Benchmark Function")
    ax.set_ylabel("Optimizer")

    plt.tight_layout()
    output_file = output_dir / f"heatmap_{dim}_{metric}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    plt.close()


def plot_box_plots(
    results: dict,
    output_dir: str | Path,
    func_name: str = "shifted_ackley",
    dim: str = "2D",
) -> None:
    """Plot box plots showing fitness distribution across runs.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
        func_name: Function name to plot
        dim: Dimensionality
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if func_name not in results["benchmarks"]:
        print(f"Warning: Function {func_name} not found in results")
        return

    func_results = results["benchmarks"][func_name].get(dim, {})

    # Collect fitness values
    data = []
    labels = []

    for optimizer_name, optimizer_data in func_results.items():
        if optimizer_data.get("runs"):
            fitness_values = [
                run["best_fitness"]
                for run in optimizer_data["runs"]
                if run["status"] == "success"
            ]
            if fitness_values:
                data.append(fitness_values)
                labels.append(optimizer_name)

    if not data:
        print(f"No data available for {func_name} ({dim})")
        return

    # Create box plot
    _fig, ax = plt.subplots(figsize=(14, 8))

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
    )

    # Color boxes
    colors = sns.color_palette("Set3", len(data))
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)

    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Best Fitness")
    ax.set_title(f"Fitness Distribution: {func_name} ({dim})")
    ax.set_yscale("log")
    ax.grid(visible=True, alpha=0.3, axis="y")

    # Rotate labels if many optimizers
    if len(labels) > 8:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    output_file = output_dir / f"boxplot_{func_name}_{dim}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved box plot to {output_file}")
    plt.close()


def plot_timing_comparison(
    results: dict, output_dir: str | Path, dim: str = "2D"
) -> None:
    """Plot timing comparison across optimizers.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
        dim: Dimensionality
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect timing data
    timing_data = {}

    for func_data in results["benchmarks"].values():
        if dim in func_data:
            for optimizer_name, optimizer_data in func_data[dim].items():
                stats = optimizer_data.get("statistics")
                if stats and stats.get("mean_time"):
                    if optimizer_name not in timing_data:
                        timing_data[optimizer_name] = []
                    timing_data[optimizer_name].append(stats["mean_time"])

    if not timing_data:
        print(f"No timing data available for {dim}")
        return

    # Calculate average time per optimizer
    avg_times = {name: np.mean(times) for name, times in timing_data.items()}

    # Sort by time
    sorted_optimizers = sorted(avg_times.items(), key=lambda x: x[1])

    names = [item[0] for item in sorted_optimizers]
    times = [item[1] for item in sorted_optimizers]

    # Create bar plot
    _fig, ax = plt.subplots(figsize=(12, 8))

    colors = sns.color_palette("viridis", len(names))
    bars = ax.barh(names, times, color=colors)

    ax.set_xlabel("Average Time (seconds)")
    ax.set_ylabel("Optimizer")
    ax.set_title(f"Runtime Comparison ({dim})")
    ax.grid(visible=True, alpha=0.3, axis="x")

    # Add value labels on bars
    for bar, time_val in zip(bars, times, strict=True):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{time_val:.3f}s",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    output_file = output_dir / f"timing_{dim}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved timing plot to {output_file}")
    plt.close()


def generate_all_plots(
    results_path: str | Path = "benchmarks/output/results.json",
    output_dir: str | Path = "benchmarks/output",
) -> None:
    """Generate all visualization plots.

    Args:
        results_path: Path to results JSON file
        output_dir: Directory to save plots
    """
    print(f"Loading results from {results_path}...")
    results = load_results(results_path)

    print("\nGenerating convergence curves...")
    for func_name in ["shifted_ackley", "rosenbrock", "rastrigin"]:
        for dim in ["2D", "10D"]:
            try:
                plot_convergence_curves(results, output_dir, func_name, dim)
            except Exception as e:
                print(f"Error plotting convergence for {func_name} {dim}: {e}")

    print("\nGenerating performance heatmaps...")
    for dim in ["2D", "10D", "30D"]:
        try:
            plot_performance_heatmap(results, output_dir, dim, "mean_fitness")
        except Exception as e:
            print(f"Error plotting heatmap for {dim}: {e}")

    print("\nGenerating box plots...")
    for func_name in ["shifted_ackley", "rosenbrock", "sphere"]:
        for dim in ["2D", "10D"]:
            try:
                plot_box_plots(results, output_dir, func_name, dim)
            except Exception as e:
                print(f"Error plotting box plot for {func_name} {dim}: {e}")

    print("\nGenerating timing comparisons...")
    for dim in ["2D", "10D", "30D"]:
        try:
            plot_timing_comparison(results, output_dir, dim)
        except Exception as e:
            print(f"Error plotting timing for {dim}: {e}")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate benchmark visualization plots"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="benchmarks/output/results.json",
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/output",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    generate_all_plots(results_path=args.results, output_dir=args.output_dir)
