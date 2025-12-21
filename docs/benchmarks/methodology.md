# Benchmarking Methodology

This page describes the scientific methodology used for benchmarking optimization algorithms in Useful Optimizer.

## Protocol Overview

Our benchmarking follows established standards from COCO and IOHprofiler:

### Run Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Independent Runs** | 30 | Statistical significance |
| **Max Evaluations** | 10,000 × dim | Standard budget |
| **Target Precision** | 1e-8 | High precision goal |
| **Dimensions Tested** | 2, 10, 30, 50 | Scalability analysis |

### Evaluation Metrics

#### Primary Metrics

1. **Best Fitness** - Final best value found
2. **Function Evaluations** - Budget used to reach target
3. **Wall Clock Time** - Computational efficiency
4. **Success Rate** - % runs reaching target

#### Statistical Metrics

1. **Mean ± Std** - Average performance with deviation
2. **Median + IQR** - Robust statistics
3. **Best/Worst** - Performance range

## Statistical Tests

### Friedman Test

For comparing multiple algorithms across multiple functions:

$$
\chi^2_F = \frac{12n}{k(k+1)} \left[ \sum_{j=1}^k R_j^2 - \frac{k(k+1)^2}{4} \right]
$$

Where:

- $n$ = number of benchmark functions
- $k$ = number of algorithms
- $R_j$ = average rank of algorithm $j$

### Wilcoxon Signed-Rank Test

For pairwise algorithm comparison:

$$
W = \sum_{i=1}^{N_r} \text{sign}(x_{2,i} - x_{1,i}) \cdot R_i
$$

### Post-hoc Nemenyi Test

Critical difference for Nemenyi test:

$$
CD = q_\alpha \sqrt{\frac{k(k+1)}{6n}}
$$

## Visualization Standards

### ECDF Curves

Empirical Cumulative Distribution Function curves show:

- **X-axis**: log₁₀(#f-evaluations / dimension)
- **Y-axis**: Proportion of targets solved
- **Multiple lines**: One per algorithm
- **Confidence bands**: 95% bootstrap CI

### Convergence Plots

Standard convergence visualization:

- **X-axis**: Iteration number or evaluations
- **Y-axis**: log₁₀(fitness - optimal)
- **Shading**: ±1 standard deviation
- **Bold line**: Median run

### Performance Profiles

Dolan-Moré performance profiles:

$$
\rho_s(\tau) = \frac{1}{|P|} \left| \{ p \in P : r_{p,s} \leq \tau \} \right|
$$

Where $r_{p,s}$ is the performance ratio.

## Benchmark Function Categories

### Unimodal Functions

- Sphere
- Rosenbrock

### Multimodal Functions

- Ackley
- Rastrigin
- Schwefel
- Griewank

### Functions with Plateaus

- Easom
- Three-hump camel

### Non-Separable Functions

- Rosenbrock
- Beale
- Goldstein-Price

## Reproducibility

### Random Seeds

All experiments use deterministic seeding:

```python
optimizer = Algorithm(
    func=benchmark_function,
    seed=42 + run_id,  # Reproducible seeds
    ...
)
```

### Hardware Configuration

Benchmarks run on standardized hardware:

- **CPU**: Intel Core i7 or equivalent
- **RAM**: 16 GB minimum
- **Python**: 3.10-3.12
- **NumPy**: >= 1.26.4

## Data Format

### IOHprofiler-Compatible JSON

```json
{
  "algorithm": "ParticleSwarm",
  "function": "sphere",
  "dimension": 10,
  "run_id": 1,
  "seed": 42,
  "trajectory": [
    {"iteration": 0, "evaluations": 50, "best_fitness": 125.3},
    {"iteration": 1, "evaluations": 100, "best_fitness": 89.2}
  ],
  "final_result": {
    "best_fitness": 1.2e-8,
    "total_evaluations": 10000,
    "wall_time_seconds": 0.42,
    "converged": true
  }
}
```

## References

1. Hansen, N., et al. "COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting." *arXiv:1603.08785* (2016).

2. Doerr, C., et al. "IOHprofiler: A Benchmarking and Profiling Tool for Iterative Optimization Heuristics." *arXiv:1810.05281* (2018).

3. Dolan, E. D., and Moré, J. J. "Benchmarking optimization software with performance profiles." *Mathematical Programming* 91.2 (2002): 201-213.
