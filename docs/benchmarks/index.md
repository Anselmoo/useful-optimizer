# Benchmarks Overview

This section provides comprehensive benchmark results comparing optimization algorithms on standard test functions, following research-grade standards inspired by COCO and IOHprofiler platforms.

## Benchmark Suite

Our benchmark suite evaluates algorithms across:

- **6 benchmark functions**: Sphere, Rosenbrock, Rastrigin, Ackley, Shifted Ackley, Griewank
- **3 dimensions**: 2D, 10D, 30D
- **30 independent runs**: Per algorithm-function-dimension combination
- **13 algorithms**: Representing all major categories

## Quick Results Summary

| Algorithm | Category | Avg. Rank | Convergence | Robustness |
|-----------|----------|-----------|-------------|------------|
| Differential Evolution | Evolutionary | 2.1 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Particle Swarm | Swarm | 2.8 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Grey Wolf Optimizer | Swarm | 3.2 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| CMA-ES | Evolutionary | 3.5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| AdamW | Gradient | 4.1 | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## Sections

### [Methodology](./methodology)

Detailed description of our benchmarking protocol:
- Test function definitions
- Parameter settings
- Statistical testing procedures
- ECDF curve generation

### [Results](./results)

Interactive visualizations including:
- ECDF curves (Empirical Cumulative Distribution Function)
- Convergence plots with confidence bands
- Violin plots for fitness distribution
- Friedman test heatmaps
- Performance profiles

### [Benchmark Functions](./functions)

Documentation of all test functions:
- Mathematical definitions
- Landscape characteristics
- Optimal solutions
- Implementation details

## Visualization Types

### ECDF Curves

The gold standard for optimizer comparison:

$$
\text{ECDF}(t) = \frac{1}{n_f \cdot n_t} \sum_{f=1}^{n_f} \sum_{i=1}^{n_t} \mathbf{1}[\text{solved}(f, t_i)]
$$

Shows the proportion of (function, target) pairs solved as a function of budget.

### Convergence Curves

Track fitness improvement over iterations with:
- Mean ± standard deviation bands
- Median with IQR shading
- Best/worst envelope

### Statistical Tests

- **Friedman Test**: Non-parametric ranking across functions
- **Wilcoxon Signed-Rank**: Pairwise statistical significance
- **Nemenyi Post-hoc**: Critical difference diagrams

## Running Benchmarks

To run the benchmark suite locally:

```bash
# Run complete benchmark suite
python benchmarks/run_benchmark_suite.py --output-dir benchmarks/output

# Generate visualization plots
python benchmarks/generate_plots.py \
    --results benchmarks/output/results.json \
    --output-dir benchmarks/output
```

## Data Format

Benchmark results are stored in IOHprofiler-compatible JSON format:

```json
{
  "metadata": {
    "max_iterations": 100,
    "n_runs": 30,
    "dimensions": [2, 10, 30],
    "timestamp": "2024-01-01 00:00:00"
  },
  "benchmarks": {
    "sphere": {
      "2D": {
        "ParticleSwarm": {
          "statistics": {
            "mean_fitness": 1.23e-5,
            "std_fitness": 2.1e-6,
            "min_fitness": 8.9e-6,
            "max_fitness": 2.1e-5,
            "median_fitness": 1.1e-5,
            "mean_time": 0.42,
            "std_time": 0.05
          },
          "success_rate": 1.0
        }
      }
    }
  }
}
```

## References

- [COCO Platform](https://github.com/numbbo/coco) - Comparing Continuous Optimizers
- [IOHprofiler](https://iohprofiler.github.io/) - Iterative Optimization Heuristics Profiler
- [IOHanalyzer](https://iohanalyzer.liacs.nl/) - Interactive performance analysis
