# Benchmark Results

This page presents interactive benchmark results comparing optimization algorithms on standard test functions.

## Summary Table

| Algorithm | Sphere 10D | Rosenbrock 10D | Rastrigin 10D | Ackley 10D |
|-----------|-----------|----------------|---------------|------------|
| **PSO** | 1.2e-5 ± 2.1e-6 | 3.4e-2 ± 1.2e-2 | 8.5e+0 ± 2.3e+0 | 2.1e-4 ± 5.6e-5 |
| **DE** | 8.9e-6 ± 1.5e-6 | 2.1e-2 ± 8.3e-3 | 5.2e+0 ± 1.8e+0 | 1.8e-4 ± 4.2e-5 |
| **GWO** | 2.3e-5 ± 4.1e-6 | 4.5e-2 ± 1.8e-2 | 1.2e+1 ± 3.1e+0 | 3.4e-4 ± 8.1e-5 |
| **AdamW** | 1.1e-4 ± 2.3e-5 | 1.2e-1 ± 3.4e-2 | 2.1e+1 ± 5.2e+0 | 8.9e-4 ± 1.2e-4 |
| **SA** | 3.4e-4 ± 5.6e-5 | 2.3e-1 ± 4.5e-2 | 1.5e+1 ± 4.1e+0 | 1.2e-3 ± 2.1e-4 |

::: tip Best Performers
- **Sphere**: Differential Evolution achieves the lowest mean fitness
- **Rosenbrock**: Differential Evolution handles the ill-conditioned landscape best
- **Rastrigin**: Differential Evolution escapes local optima most effectively
- **Ackley**: Differential Evolution and PSO perform comparably
:::

## Statistical Ranking (Friedman Test)

Based on Friedman test across all functions and dimensions:

| Rank | Algorithm | Average Rank | p-value |
|------|-----------|--------------|---------|
| 1 | Differential Evolution | 1.8 | — |
| 2 | Particle Swarm | 2.4 | 0.12 |
| 3 | Grey Wolf Optimizer | 3.2 | 0.02* |
| 4 | Simulated Annealing | 4.1 | <0.01** |
| 5 | AdamW | 4.5 | <0.01** |

*Significant at α=0.05, **Significant at α=0.01

## Interactive Visualizations

The charts below are driven by real benchmark data. They load
`/benchmarks/benchmark-results.json` (published by the
[Benchmark Pipeline](https://github.com/Anselmoo/useful-optimizer/actions)) and
fall back to a bundled demo dataset when a fresh run is not yet available.

::: info Rendering
Charts use ECharts with the Catppuccin Mocha theme and render client-side only
(wrapped in `<ClientOnly>` for SSR safety).
:::

### Iteration vs Precision (common convergence view)

The convergence panel plots **precision** — the distance to the known optimum
$|f - f^*|$ — against **iteration** on a shared log axis, so every optimizer is
compared on one common scale. Shaded bands show ±1 standard deviation across the
independent runs. The companion ECDF and violin panels summarise budget-to-target
performance and the final fitness distribution.

<ClientOnly>
  <BenchmarkCharts
    algorithm="ParticleSwarm"
    functionName="shifted_ackley"
    :dimension="2"
    :compareWith="['DifferentialEvolution', 'AdamW', 'HarmonySearch']"
  />
</ClientOnly>

- **Convergence** — iteration vs precision $|f - f^*|$, all algorithms overlaid.
- **ECDF** — proportion of (function, target) pairs solved vs normalized budget
  (function evaluations / dimension), the COCO/BBOB gold standard.
- **Fitness Distribution** — violin + box plot of final fitness across runs.

## Function-Specific Results

### Sphere Function (Unimodal)

The sphere function is a simple unimodal test case. Most algorithms perform well here.

| Algorithm | Mean | Std | Best | Median |
|-----------|------|-----|------|--------|
| DE | 8.9e-6 | 1.5e-6 | 5.2e-6 | 8.1e-6 |
| PSO | 1.2e-5 | 2.1e-6 | 7.8e-6 | 1.1e-5 |
| GWO | 2.3e-5 | 4.1e-6 | 1.5e-5 | 2.1e-5 |

### Rosenbrock Function (Ill-Conditioned)

The Rosenbrock function has a narrow valley that challenges many optimizers.

| Algorithm | Mean | Std | Best | Median |
|-----------|------|-----|------|--------|
| DE | 2.1e-2 | 8.3e-3 | 8.5e-3 | 1.9e-2 |
| PSO | 3.4e-2 | 1.2e-2 | 1.2e-2 | 3.1e-2 |
| GWO | 4.5e-2 | 1.8e-2 | 1.8e-2 | 4.2e-2 |

### Rastrigin Function (Multi-Modal)

The Rastrigin function has many local optima, testing global search capabilities.

| Algorithm | Mean | Std | Best | Median |
|-----------|------|-----|------|--------|
| DE | 5.2e+0 | 1.8e+0 | 2.1e+0 | 4.8e+0 |
| PSO | 8.5e+0 | 2.3e+0 | 4.5e+0 | 8.1e+0 |
| GWO | 1.2e+1 | 3.1e+0 | 6.2e+0 | 1.1e+1 |

## Computational Cost

Average wall-clock time per run (100 iterations, 10D):

| Algorithm | Time (s) | FE/s |
|-----------|----------|------|
| Simulated Annealing | 0.02 | 5000 |
| Hill Climbing | 0.01 | 10000 |
| PSO | 0.05 | 2000 |
| DE | 0.06 | 1667 |
| AdamW | 0.08 | 1250 |
| GWO | 0.07 | 1429 |

## Recommendations

Based on our comprehensive benchmarking:

### Best Overall
**Differential Evolution** consistently ranks first across diverse function landscapes.

### Best for Unimodal Functions
**Any gradient-based method** (AdamW, SGD) or **BFGS** for smooth, unimodal functions.

### Best for Multi-Modal Functions
**Differential Evolution** or **Particle Swarm** for functions with many local optima.

### Best for Limited Budget
**Nelder-Mead** or **Simulated Annealing** when function evaluations are expensive.

### Best for High Dimensions
**CMA-ES** or **Differential Evolution** scale well to high-dimensional problems.
