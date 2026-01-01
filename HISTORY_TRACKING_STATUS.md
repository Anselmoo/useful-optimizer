# History Tracking Implementation Status

## Overview
This document tracks the implementation of history tracking across all optimizers for COCO/BBOB benchmark compliance.

## Completed Infrastructure ✅

### 1. Core Infrastructure
- [x] `opt/abstract/history.py` - Memory-efficient `OptimizationHistory` class with pre-allocated NumPy arrays
- [x] `opt/abstract/` module structure created
- [x] `docs/schemas/docstring-schema.json` updated with new property enums
- [x] `.pre-commit-config.yaml` updated to exclude `__init__.py` files
- [x] `benchmarks/generate_plots.py` fixed to use correct history access pattern

### 2. Abstract Base Classes
- [x] `opt/abstract/single_objective.py` (copy of `abstract_optimizer.py`)
- [x] `opt/abstract/multi_objective.py` (copy of `abstract_multi_objective.py`)
- [x] `opt/abstract/__init__.py` with re-exports

### 3. Helper Scripts
- [x] `scripts/add_history_tracking_to_optimizer.py` - Template and instructions

## Optimizers with History Tracking ✅ (3/120+)

### Benchmark Suite Optimizers (13 total, 3 implemented)
- [x] **ParticleSwarm** (`opt/swarm_intelligence/particle_swarm.py`) - COMPLETE
- [x] **AntColony** (`opt/swarm_intelligence/ant_colony.py`) - COMPLETE
- [x] **FireflyAlgorithm** (`opt/swarm_intelligence/firefly_algorithm.py`) - COMPLETE
- [ ] BatAlgorithm (`opt/swarm_intelligence/bat_algorithm.py`)
- [ ] GreyWolfOptimizer (`opt/swarm_intelligence/grey_wolf_optimizer.py`)
- [ ] GeneticAlgorithm (`opt/evolutionary/genetic_algorithm.py`)
- [ ] DifferentialEvolution (`opt/evolutionary/differential_evolution.py`)
- [ ] HarmonySearch (`opt/metaheuristic/harmony_search.py`)
- [ ] SimulatedAnnealing (`opt/classical/simulated_annealing.py`)
- [ ] HillClimbing (`opt/classical/hill_climbing.py`)
- [ ] NelderMead (`opt/classical/nelder_mead.py`)
- [ ] AdamW (`opt/gradient_based/adamw.py`)
- [ ] SGDMomentum (`opt/gradient_based/sgd_momentum.py`)

## Implementation Pattern

### Required Changes per Optimizer

#### 1. Add `track_history` parameter to `__init__`:
```python
def __init__(
    self,
    func: Callable[[ndarray], float],
    # ... other parameters ...
    seed: int | None = None,
    track_history: bool = False,  # ADD THIS
) -> None:
```

#### 2. Pass `track_history` to `super().__init__`:
```python
super().__init__(
    func=func,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    dim=dim,
    max_iter=max_iter,
    seed=seed,
    population_size=population_size,
    track_history=track_history,  # ADD THIS
)
```

#### 3. Record history in `search()` method:
```python
def search(self) -> tuple[np.ndarray, float]:
    # ... initialization ...

    for iteration in range(self.max_iter):
        # Track history at START of iteration
        if self.track_history:
            self.history["best_fitness"].append(float(best_fitness))
            self.history["best_solution"].append(best_solution.copy())

        # ... optimization logic ...

    # Track final state before return
    if self.track_history:
        self.history["best_fitness"].append(float(best_fitness))
        self.history["best_solution"].append(best_solution.copy())

    return best_solution, best_fitness
```

## Validation Errors Remaining

### Schema Validation Issues (~100 files)
Many files have invalid `properties` values that aren't in the schema enum. Common issues:
- "Nature-inspired" (added to schema but files not yet updated)
- "Bio-inspired" (added to schema but files not yet updated)
- "Swarm-based" (added to schema but files not yet updated)
- Custom properties like "Chain formation", "Parameter-free", etc.

**Note:** These don't block history tracking implementation and can be fixed separately.

## Testing

### Verify History Tracking Works:
```python
from benchmarks.run_benchmark_suite import run_single_benchmark
from opt.swarm_intelligence.ant_colony import AntColony
from opt.benchmark.functions import shifted_ackley

result = run_single_benchmark(
    AntColony,
    shifted_ackley,
    -32.768, 32.768,
    dim=2,
    max_iter=50,
    seed=42
)

print('Status:', result.get('status'))
print('History length:', len(result.get('convergence_history', [])))
# Expected output: Status: success, History length: 51 (max_iter + 1)
```

## Next Steps

### Priority 1: Complete Benchmark Suite Optimizers
Add history tracking to the remaining 10 optimizers used in `benchmarks/run_benchmark_suite.py`:
1. BatAlgorithm
2. GreyWolfOptimizer
3. GeneticAlgorithm
4. DifferentialEvolution
5. HarmonySearch
6. SimulatedAnnealing
7. HillClimbing
8. NelderMead
9. AdamW
10. SGDMomentum

### Priority 2: Run Full Benchmark Suite
Once all benchmark optimizers have history tracking:
```bash
cd benchmarks
uv run python run_benchmark_suite.py
```

### Priority 3: Remaining Optimizers (~107 files)
Add history tracking to all other optimizers in:
- `opt/swarm_intelligence/` (~50 files)
- `opt/evolutionary/` (~6 files)
- `opt/metaheuristic/` (~14 files)
- `opt/classical/` (~9 files)
- `opt/gradient_based/` (~11 files)
- `opt/physics_inspired/` (~4 files)
- `opt/probabilistic/` (~5 files)
- `opt/social_inspired/` (~3 files)
- `opt/constrained/` (~5 files)

### Priority 4: Fix Schema Validation Errors
Update ~100 optimizer files with invalid property values to match schema enums.

## Memory Efficiency Notes

The `OptimizationHistory` class in `opt/abstract/history.py` provides:
- **O(1) recording operations** (vs O(n) for list.append amortized)
- **Pre-allocated NumPy arrays** for contiguous memory
- **Configurable tracking** via `HistoryConfig` dataclass
- **IOHprofiler-compatible export** via `to_dict()`

For 10,000 iterations, dim=30, population=100:
- best_fitness: 80 KB
- best_solution: 2.4 MB
- population: 240 MB (only if tracked)

## References
- Issue: [#111 COCO/BBOB Benchmark Data Collection Suite](https://github.com/Anselmoo/useful-optimizer/issues/111)
- COCO Platform: https://coco-platform.org/
- IOHprofiler: https://iohprofiler.github.io/
