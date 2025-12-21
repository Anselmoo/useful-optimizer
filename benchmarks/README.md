# Benchmark Suite

This directory contains automated benchmark scripts for comparing optimization algorithms across standard benchmark functions.

## Overview

The benchmark suite evaluates all optimization algorithms in the `useful-optimizer` package on multiple benchmark functions across different dimensionalities.

## Files

- **`run_benchmark_suite.py`**: Main benchmark runner that executes all optimizers on benchmark functions
- **`generate_plots.py`**: Visualization generator that creates plots from benchmark results
- **`output/`**: Directory for generated results and plots (gitignored)

## Usage

### Running Benchmarks

Run the complete benchmark suite:

```bash
uv run python benchmarks/run_benchmark_suite.py
```

This will:
- Test 13 optimizers on 6 benchmark functions
- Run across 3 dimensionalities (2D, 10D, 30D)
- Execute 10 runs per configuration for statistical reliability
- Save results to `benchmarks/output/results.json`

### Generating Visualizations

After running benchmarks, generate plots:

```bash
uv run python benchmarks/generate_plots.py
```

This creates:
- **Convergence curves**: Fitness vs. iteration for each optimizer
- **Performance heatmaps**: Algorithm × Function matrix with color-coded performance
- **Box plots**: Distribution of final fitness across multiple runs
- **Timing comparisons**: Runtime benchmarks across optimizers

All plots are saved to `benchmarks/output/` as high-resolution PNG files (300 DPI).

### Custom Configuration

Specify custom output directory:

```bash
uv run python benchmarks/run_benchmark_suite.py --output-dir my_results
uv run python benchmarks/generate_plots.py --results my_results/results.json --output-dir my_results
```

## Benchmark Functions

The suite includes:

1. **Sphere**: Simple unimodal function
2. **Rosenbrock**: Valley-shaped function
3. **Rastrigin**: Highly multimodal function
4. **Ackley**: Multimodal with many local minima
5. **Shifted Ackley**: Non-centered variant
6. **Griewank**: Product and sum of coordinates

## Tested Optimizers

### Swarm Intelligence
- ParticleSwarm
- AntColony
- FireflyAlgorithm
- BatAlgorithm
- GreyWolfOptimizer

### Evolutionary
- GeneticAlgorithm
- DifferentialEvolution

### Metaheuristic
- HarmonySearch
- SimulatedAnnealing

### Classical
- HillClimbing
- NelderMead

### Gradient-Based
- AdamW
- SGDMomentum

## Configuration

Default settings (can be modified in `run_benchmark_suite.py`):

- **Dimensions**: 2D, 10D, 30D
- **Max iterations**: 100
- **Runs per configuration**: 10
- **Population size**: 30 (where applicable)
- **Random seeds**: 42-51 for reproducibility

## CI/CD Integration

The benchmark suite runs automatically via GitHub Actions:

- **Schedule**: Weekly on Sunday at 00:00 UTC
- **Triggers**: Manual workflow dispatch, or changes to benchmark/optimizer code
- **Artifacts**: Results and visualizations uploaded with 90-day retention

To trigger manually:
1. Go to Actions → Benchmark Visualizations
2. Click "Run workflow"
3. Download artifacts after completion

## Output Format

### Results JSON Structure

```json
{
  "metadata": {
    "max_iterations": 100,
    "n_runs": 10,
    "dimensions": [2, 10, 30],
    "timestamp": "2024-12-21 16:00:00"
  },
  "benchmarks": {
    "function_name": {
      "2D": {
        "OptimizerName": {
          "runs": [...],
          "statistics": {
            "mean_fitness": 0.001,
            "std_fitness": 0.0002,
            "min_fitness": 0.0008,
            "max_fitness": 0.0015,
            "median_fitness": 0.001,
            "mean_time": 0.5,
            "std_time": 0.05
          },
          "success_rate": 1.0
        }
      }
    }
  }
}
```

## Visualization Types

### 1. Convergence Curves
Shows best fitness over iterations for each optimizer on a specific function.

**Files**: `convergence_{function}_{dim}.png`

### 2. Performance Heatmaps
Matrix visualization of optimizer performance across all functions.

**Files**: `heatmap_{dim}_{metric}.png`

### 3. Box Plots
Statistical distribution of fitness values across multiple runs.

**Files**: `boxplot_{function}_{dim}.png`

### 4. Timing Comparisons
Horizontal bar chart comparing average runtime per optimizer.

**Files**: `timing_{dim}.png`

## Dependencies

The benchmark suite requires additional dependencies:

```bash
uv sync --extra benchmark
```

This installs:
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Tips

1. **Quick Test**: Reduce `MAX_ITERATIONS` and `N_RUNS` in `run_benchmark_suite.py` for faster testing
2. **Focus on Specific Functions**: Comment out unwanted functions in `BENCHMARK_FUNCTIONS`
3. **Add Optimizers**: Import and add to `OPTIMIZERS` dict in `run_benchmark_suite.py`
4. **Custom Plots**: Modify `generate_plots.py` to create additional visualizations

## Performance Notes

- **Full suite runtime**: ~30-45 minutes (780 total configurations)
- **Reduced suite** (fewer runs/iterations): ~5-10 minutes
- **Memory usage**: ~500MB peak
- **Output size**: ~50-100MB (JSON + all plots)

## Troubleshooting

**Issue**: Some optimizers fail on certain functions
- **Solution**: This is expected. The suite tracks success rate per optimizer

**Issue**: Memory error on high dimensions
- **Solution**: Reduce population sizes or max iterations for 30D tests

**Issue**: Plots look cluttered
- **Solution**: Adjust figure sizes in `generate_plots.py` or filter to fewer optimizers

## Contributing

To add new benchmark functions:

1. Add function to `opt/benchmark/functions.py`
2. Import in `run_benchmark_suite.py`
3. Add to `BENCHMARK_FUNCTIONS` dict with appropriate bounds

To add new optimizers:

1. Import optimizer class
2. Add to `OPTIMIZERS` dict
3. Ensure it follows `AbstractOptimizer` interface

## References

Inspired by:
- [scikit-learn benchmarks](https://github.com/scikit-learn/scikit-learn/tree/main/benchmarks)
- [scikit-learn examples gallery](https://scikit-learn.org/stable/auto_examples/index.html)
