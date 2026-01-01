# COCO/BBOB Benchmark Data Pipeline

This implementation provides a complete, COCO/BBOB-compliant benchmark data pipeline for the useful-optimizer library.

## Overview

The pipeline consists of 5 main components:

1. **AbstractOptimizer Extensions** - Early stopping and evaluation tracking
2. **Function Optima Mapping** - Known optimal values for benchmark functions
3. **Benchmark Runner** - BBOB-compliant test execution
4. **Aggregation Script** - Statistical processing and ERT calculation
5. **CI Workflow** - Automated weekly benchmarking with artifact storage

## Quick Start

### Run Benchmarks Locally

```bash
# Quick test with 4 showcase optimizers (PSO, DE, AdamW, HS)
uv run python benchmarks/run_benchmark_suite.py --subset

# Full benchmark with 13 optimizers (takes longer)
uv run python benchmarks/run_benchmark_suite.py

# Custom output directory
uv run python benchmarks/run_benchmark_suite.py --output-dir my_results/
```

### Process Results

```bash
# Aggregate raw results into statistical summary
uv run python benchmarks/aggregate_results.py \
  --input benchmarks/output/results.json \
  --output-dir benchmarks/output/processed \
  --validate
```

## Architecture

### Evaluation Tracking

All optimizers now track:
- `n_evaluations` - Total function evaluations
- `converged` - Whether target precision was reached
- `evaluations_to_target` - Evaluations when first converged

Example usage:

```python
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.benchmark.functions import sphere

pso = ParticleSwarm(
    func=sphere,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    max_iter=100,
    target_precision=1e-8,  # BBOB standard
    f_opt=0.0,              # Known optimum for sphere
)

best_solution, best_fitness = pso.search()

print(f"Evaluations: {pso.n_evaluations}")
print(f"Converged: {pso.converged}")
print(f"Evals to target: {pso.evaluations_to_target}")
```

### Function Optima

Use the `optima.py` module to get known optimal values:

```python
from opt.benchmark.optima import get_optimum, is_converged

# Get known optimum
f_opt = get_optimum('sphere')  # Returns 0.0

# Check convergence
converged = is_converged(
    current_fitness=1e-9,
    func_name='sphere',
    target_precision=1e-8
)  # Returns True
```

### Benchmark Configuration

Current settings (in `run_benchmark_suite.py`):

```python
DIMENSIONS = [2, 5, 10, 20]  # BBOB standard dimensions
MAX_ITERATIONS = 1000        # Per optimizer run
N_RUNS = 15                  # BBOB requires 15 independent runs
TARGET_PRECISION = 1e-8      # BBOB standard

# Showcase optimizers for CI
SHOWCASE_OPTIMIZERS = {
    "ParticleSwarm": ParticleSwarm,
    "DifferentialEvolution": DifferentialEvolution,
    "AdamW": AdamW,
    "HarmonySearch": HarmonySearch,
}
```

### Aggregation Output

The aggregation script produces:

```json
{
  "metadata": {
    "max_iterations": 1000,
    "n_runs": 15,
    "dimensions": [2, 5, 10, 20],
    "timestamp": "2026-01-01T16:00:00+00:00",
    "python_version": "3.12.3",
    "numpy_version": "2.2.1"
  },
  "benchmarks": {
    "sphere": {
      "2D": {
        "ParticleSwarm": {
          "runs": [...],  // Individual run data
          "statistics": {
            "mean_fitness": 0.001234,
            "std_fitness": 0.000567,
            "min_fitness": 0.000123,
            "max_fitness": 0.003456,
            "median_fitness": 0.001111,
            "q1_fitness": 0.000789,
            "q3_fitness": 0.001567,
            "ert": 123.45  // Expected Running Time
          },
          "success_rate": 0.8  // 12 out of 15 runs converged
        }
      }
    }
  }
}
```

## CI/CD Integration

### Workflow Triggers

- **Scheduled**: Every Sunday at 2:00 AM UTC
- **Manual**: Via GitHub Actions UI with subset option

### Artifacts

The pipeline uploads two artifacts:

1. **benchmark-summary-{sha}** (90 days retention)
   - Processed statistical summaries
   - ~2-5 MB (within GitHub Free tier)
   - Schema-validated JSON

2. **benchmark-raw-{sha}** (30 days retention)
   - Raw benchmark results
   - Full convergence histories
   - Larger file size

### Manual Trigger

```yaml
# Go to Actions > Benchmark Pipeline > Run workflow
# Options:
#   - subset: true  (4 optimizers, ~15 min)
#   - subset: false (13 optimizers, ~60 min)
```

## Data Schema

Results follow `docs/schemas/benchmark-data-schema.json`:

- **IOHprofiler-compatible** format
- **Pydantic validation** via `benchmarks/models.py`
- **COCO/BBOB standard** metrics

## Performance Metrics

### Expected Running Time (ERT)

ERT measures the average number of evaluations needed to reach target precision:

```
ERT = (total_evaluations) / (successful_runs / total_runs)
```

Example:
- 15 runs total
- 12 runs converged to 1e-8 precision
- Average evaluations for successful runs: 500
- ERT = 500 / (12/15) = 625

### Success Rate

Percentage of runs that reached target precision:

```
success_rate = successful_runs / total_runs
```

## Adding New Optimizers

To add a new optimizer to the benchmark:

1. Ensure it inherits from `AbstractOptimizer`
2. Add `**kwargs` to `__init__` and `super().__init__()`:

```python
def __init__(
    self,
    func: Callable,
    lower_bound: float,
    upper_bound: float,
    dim: int,
    max_iter: int = 1000,
    seed: int | None = None,
    **kwargs,  # Accept COCO/BBOB parameters
) -> None:
    super().__init__(
        func=func,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        dim=dim,
        max_iter=max_iter,
        seed=seed,
        **kwargs,  # Pass through to base class
    )
    # Your optimizer-specific initialization
```

3. Add to `OPTIMIZERS` dict in `run_benchmark_suite.py`

## Troubleshooting

### "TypeError: unexpected keyword argument"

Make sure the optimizer accepts `**kwargs` in its `__init__` method.

### "KeyError: function not found"

Add the function optimum to `opt/benchmark/optima.py`:

```python
FUNCTION_OPTIMA = {
    "my_function": 0.0,  # Add your function
    # ...
}
```

### Schema Validation Fails

Ensure your results match `benchmark-data-schema.json`:
- All required fields present
- Correct data types
- Valid ranges (e.g., success_rate ∈ [0, 1])

## References

- [COCO Platform](https://github.com/numbbo/coco) - Comparing Continuous Optimizers
- [IOHprofiler Data Format](https://iohprofiler.github.io/IOHexp/data/)
- [BBOB Function Documentation](https://coco.gforge.inria.fr/downloads/download16.00/bbobdocfunctions.pdf)
- [GitHub Actions Artifacts](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts)

## License

MIT - Same as useful-optimizer
