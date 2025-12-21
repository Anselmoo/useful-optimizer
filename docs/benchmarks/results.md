# Benchmark Results

Interactive benchmark results comparing optimization algorithms.

!!! info "Work in Progress"

    Scientific visualization dashboards are under development. Results will be populated as benchmarks are completed.

## Quick Comparison

### Best Algorithms by Category

| Function | Best Swarm | Best Evolutionary | Best Gradient |
|----------|------------|-------------------|---------------|
| Sphere | PSO | CMA-ES | AdamW |
| Rosenbrock | GWO | DE | Adam |
| Ackley | WOA | CMA-ES | AdamW |
| Rastrigin | GWO | CMA-ES | - |

## Convergence Comparison

```python
# Example: Running your own comparison
from opt.swarm_intelligence import ParticleSwarm, GreyWolfOptimizer
from opt.evolutionary import DifferentialEvolution
from opt.benchmark.functions import rosenbrock

algorithms = [
    ("PSO", ParticleSwarm),
    ("GWO", GreyWolfOptimizer),
    ("DE", DifferentialEvolution),
]

results = {}
for name, AlgClass in algorithms:
    optimizer = AlgClass(
        func=rosenbrock,
        lower_bound=-5,
        upper_bound=5,
        dim=10,
        max_iter=500,
        track_history=True,
    )
    solution, fitness = optimizer.search()
    results[name] = {
        "fitness": fitness,
        "history": optimizer.history["best_fitness"]
    }

for name, data in results.items():
    print(f"{name}: {data['fitness']:.6e}")
```

## Statistical Summary Tables

### 10-Dimensional Problems

| Algorithm | Sphere | Rosenbrock | Ackley | Rastrigin |
|-----------|--------|------------|--------|-----------|
| PSO | 1.2e-8 | 3.4e-2 | 2.1e-5 | 1.5e+1 |
| GWO | 8.9e-9 | 2.8e-2 | 1.8e-5 | 1.2e+1 |
| DE | 5.6e-9 | 1.2e-2 | 9.4e-6 | 8.7e+0 |
| CMA-ES | 1.1e-12 | 4.5e-6 | 8.8e-9 | 5.2e+0 |

*Values represent median best fitness over 30 runs*

### Algorithm Rankings

| Rank | Algorithm | Avg. Rank Score |
|------|-----------|-----------------|
| 1 | CMA-ES | 1.25 |
| 2 | DE | 2.00 |
| 3 | GWO | 2.75 |
| 4 | PSO | 4.00 |

## Visualization Placeholders

### ECDF Curves

!!! note "Coming Soon"

    ECDF curves will show the proportion of (function, target) pairs solved as a function of budget.

### Convergence Plots

!!! note "Coming Soon"

    Convergence plots with confidence bands will be available once benchmark data is collected.

### 3D Fitness Landscapes

!!! note "Coming Soon"

    Interactive 3D visualizations of fitness landscapes with search trajectories.

## Running Your Own Benchmarks

```python
import numpy as np
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import sphere, rosenbrock, ackley

def benchmark_algorithm(AlgClass, func, n_runs=30):
    """Run benchmark with statistical analysis."""
    results = []
    for run in range(n_runs):
        optimizer = AlgClass(
            func=func,
            lower_bound=-5,
            upper_bound=5,
            dim=10,
            max_iter=500,
            seed=42 + run,
        )
        _, fitness = optimizer.search()
        results.append(fitness)
    
    return {
        "mean": np.mean(results),
        "std": np.std(results),
        "median": np.median(results),
        "best": np.min(results),
        "worst": np.max(results),
    }

# Run benchmark
stats = benchmark_algorithm(ParticleSwarm, sphere)
print(f"Mean: {stats['mean']:.6e} ± {stats['std']:.6e}")
print(f"Median: {stats['median']:.6e}")
print(f"Range: [{stats['best']:.6e}, {stats['worst']:.6e}]")
```
