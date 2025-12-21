# Benchmark Methodology

This document describes the rigorous methodology used for benchmarking optimization algorithms in Useful Optimizer.

## Protocol Overview

Our benchmarking follows established standards from the optimization research community, particularly COCO (Comparing Continuous Optimizers) and IOHprofiler platforms.

### Key Principles

1. **Reproducibility**: Fixed random seeds for each run
2. **Statistical Validity**: 30 independent runs per configuration
3. **Fair Comparison**: Same budget (function evaluations) for all algorithms
4. **Comprehensive Testing**: Multiple functions, dimensions, and metrics

## Test Functions

### Sphere Function

$$
f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2
$$

- **Optimum**: $f(\mathbf{0}) = 0$
- **Bounds**: $[-5.12, 5.12]^n$
- **Characteristics**: Unimodal, separable, convex

### Rosenbrock Function

$$
f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]
$$

- **Optimum**: $f(\mathbf{1}) = 0$
- **Bounds**: $[-5, 10]^n$
- **Characteristics**: Unimodal, non-separable, ill-conditioned

### Rastrigin Function

$$
f(\mathbf{x}) = 10n + \sum_{i=1}^{n} \left[ x_i^2 - 10\cos(2\pi x_i) \right]
$$

- **Optimum**: $f(\mathbf{0}) = 0$
- **Bounds**: $[-5.12, 5.12]^n$
- **Characteristics**: Multi-modal, separable

### Ackley Function

$$
f(\mathbf{x}) = -20\exp\left(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}\right) - \exp\left(\frac{1}{n}\sum_{i=1}^{n}\cos(2\pi x_i)\right) + 20 + e
$$

- **Optimum**: $f(\mathbf{0}) = 0$
- **Bounds**: $[-32.768, 32.768]^n$
- **Characteristics**: Multi-modal, nearly flat outer region

### Griewank Function

$$
f(\mathbf{x}) = \sum_{i=1}^{n} \frac{x_i^2}{4000} - \prod_{i=1}^{n} \cos\left(\frac{x_i}{\sqrt{i}}\right) + 1
$$

- **Optimum**: $f(\mathbf{0}) = 0$
- **Bounds**: $[-600, 600]^n$
- **Characteristics**: Multi-modal, many local minima

## Experimental Setup

### Parameters

| Parameter | Value |
|-----------|-------|
| Dimensions | 2, 10, 30 |
| Independent runs | 30 |
| Maximum iterations | 100 |
| Population size | 30 (where applicable) |
| Random seeds | 42, 43, ..., 71 |

### Algorithm Settings

All algorithms use their default parameters as defined in the library, with the following exceptions:
- Population-based algorithms use `population_size=30`
- All algorithms use `max_iter=100`

## Statistical Analysis

### Friedman Test

Non-parametric test for comparing multiple algorithms across multiple functions:

$$
\chi_F^2 = \frac{12N}{k(k+1)} \left[ \sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4} \right]
$$

where $N$ is the number of functions, $k$ is the number of algorithms, and $R_j$ is the average rank of algorithm $j$.

### Wilcoxon Signed-Rank Test

Pairwise comparison with Bonferroni correction:

$$
\alpha_{\text{adjusted}} = \frac{\alpha}{m}
$$

where $m$ is the number of pairwise comparisons.

### ECDF Calculation

For a set of runs, the ECDF at budget $B$ is:

$$
\text{ECDF}(B) = \frac{\text{Number of runs reaching target at budget} \leq B}{\text{Total number of runs}}
$$

Target precisions: $10^{-1}, 10^{-3}, 10^{-5}, 10^{-7}$

## Metrics Reported

### Primary Metrics

- **Best Fitness**: Minimum fitness achieved
- **Mean Fitness**: Average across 30 runs
- **Std Fitness**: Standard deviation across runs
- **Success Rate**: Proportion of runs reaching target

### Secondary Metrics

- **Median Fitness**: Robust central tendency
- **Mean Time**: Average wall-clock time
- **Function Evaluations**: Number of objective function calls

## Reproducibility

All benchmark results can be reproduced using:

```bash
# Set up environment
uv sync --all-extras

# Run benchmarks
uv run python benchmarks/run_benchmark_suite.py \
    --output-dir benchmarks/output

# Generate visualizations
uv run python benchmarks/generate_plots.py \
    --results benchmarks/output/results.json \
    --output-dir benchmarks/output
```

## References

1. Hansen, N., et al. "COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting." arXiv:1603.08785 (2016).

2. Doerr, C., et al. "IOHprofiler: A Benchmarking and Profiling Tool for Iterative Optimization Heuristics." arXiv:1810.05281 (2018).

3. Demšar, J. "Statistical Comparisons of Classifiers over Multiple Data Sets." JMLR 7 (2006): 1-30.
