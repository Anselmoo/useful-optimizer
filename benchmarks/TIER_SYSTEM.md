# Benchmark Tier System

## Overview

The benchmark suite uses a **tiered optimizer selection system** to balance comprehensive coverage with practical runtime constraints.

## Why Not Test All 120+ Optimizers?

**Simple answer**: It's computationally prohibitive for CI/CD.

### The Math

| Tier | Optimizers | Runs | Runtime | Artifact Size | Use Case |
|------|-----------|------|---------|---------------|----------|
| **Showcase** | 4 | 1,440 | ~15-30 min | ~5-10 MB | ✅ PR checks, quick validation |
| **Standard** | 13 | 4,680 | ~6-8 hours | ~15-30 MB | ✅ Daily/weekly comprehensive testing |
| **Comprehensive** | 120+ | 43,200+ | ~55-70 hours | ~500MB-1GB | ❌ Exceeds GitHub CI limits |

**Calculation**: `n_optimizers × 6 functions × 4 dimensions × 15 runs`

### CI/CD Constraints

- **GitHub Actions free tier**: 6 hours max per job
- **Artifact size limit**: 10GB total per repo (comprehensive tier = 1GB per run)
- **Cost**: Comprehensive tier would consume 10-12 hours of billable time per run

## Tier Definitions

### 🚀 Showcase (Fast)
**Purpose**: PR validation, quick smoke tests
**Optimizers**: 4 representative algorithms (one per category)
**Runtime**: 15-30 minutes
**Coverage**: Basic validation across major algorithm families

```bash
uv run python benchmarks/run_benchmark_suite.py --tier showcase
```

**Algorithms**:
- `ParticleSwarm` (Swarm Intelligence)
- `DifferentialEvolution` (Evolutionary)
- `AdamW` (Gradient-based)
- `HarmonySearch` (Metaheuristic)

---

### 📊 Standard (Balanced)
**Purpose**: Daily/weekly comprehensive testing
**Optimizers**: 13 algorithms across all categories
**Runtime**: 6-8 hours
**Coverage**: Representative sample ensuring statistical significance

```bash
uv run python benchmarks/run_benchmark_suite.py --tier standard
```

**Algorithms** (by category):
- **Swarm Intelligence**: ParticleSwarm, AntColony, FireflyAlgorithm, BatAlgorithm, GreyWolfOptimizer
- **Evolutionary**: GeneticAlgorithm, DifferentialEvolution
- **Metaheuristic**: HarmonySearch, SimulatedAnnealing
- **Classical**: HillClimbing, NelderMead
- **Gradient-based**: AdamW, SGDMomentum

---

### 🔬 Comprehensive (Exhaustive)
**Purpose**: Monthly/quarterly deep analysis
**Optimizers**: All 120+ algorithms
**Runtime**: 55-70 hours
**Status**: ⚠️ **NOT YET IMPLEMENTED** (requires dynamic loading)

```bash
# Not yet available
uv run python benchmarks/run_benchmark_suite.py --tier comprehensive
# Error: NotImplementedError - use standard tier for now
```

**Blocker**: Requires dynamic discovery and loading of all optimizer classes from `opt/*` directories.

---

## Recommended Workflow

### For CI/CD Pipelines

```yaml
# .github/workflows/benchmark-quick.yml (on PR)
- name: Quick Benchmark Validation
  run: uv run python benchmarks/run_benchmark_suite.py --tier showcase

# .github/workflows/benchmark-weekly.yml (scheduled)
- name: Weekly Comprehensive Benchmark
  run: uv run python benchmarks/run_benchmark_suite.py --tier standard
```

### For Local Development

```bash
# Quick sanity check (before committing)
uv run python benchmarks/run_benchmark_suite.py --tier showcase --output-dir /tmp/bench

# Full validation (before major release)
uv run python benchmarks/run_benchmark_suite.py --tier standard
```

### For Research Analysis

```bash
# Standard tier with custom output
uv run python benchmarks/run_benchmark_suite.py \
  --tier standard \
  --output-dir ~/research/benchmarks/$(date +%Y%m%d)

# Process results
uv run python benchmarks/aggregate_results.py \
  --input ~/research/benchmarks/$(date +%Y%m%d)/results.json \
  --output-dir ~/research/processed
```

---

## Backward Compatibility

The old `--subset` flag is still supported but deprecated:

```bash
# OLD (deprecated but works)
uv run python benchmarks/run_benchmark_suite.py --subset

# NEW (preferred)
uv run python benchmarks/run_benchmark_suite.py --tier showcase
```

**Warning message**: `--subset flag is deprecated. Use --tier showcase/standard/comprehensive instead.`

---

## Future Work: Comprehensive Tier

To implement the comprehensive tier (all 120+ optimizers):

1. **Dynamic optimizer discovery**:
   ```python
   def discover_all_optimizers():
       """Scan opt/* directories and import all optimizer classes."""
       # Scan opt/classical/, opt/evolutionary/, etc.
       # Filter for AbstractOptimizer subclasses
       # Return dict of {name: class}
   ```

2. **Chunked execution** (to work within CI limits):
   ```bash
   # Run in 10-hour chunks over 6 days
   for chunk in {1..6}; do
     uv run python benchmarks/run_benchmark_suite.py \
       --tier comprehensive --chunk $chunk/6
   done
   ```

3. **Distributed benchmarking** (future):
   - Split by function or dimension
   - Parallel execution across multiple workers
   - Aggregate results at the end

---

## Statistical Validity

All tiers follow **BBOB/COCO standards**:
- ✅ 15 independent runs per configuration
- ✅ Multiple dimensions: [2, 5, 10, 20]
- ✅ Target precision: 1e-8
- ✅ Reproducible seeds: 0-14

Even the **showcase tier** (4 optimizers) provides statistically significant results for quick validation.

---

## Questions?

- **Why 13 optimizers in standard tier?** Representative sample across all 10 algorithm categories
- **Can I add custom tiers?** Yes - edit `OPTIMIZER_TIERS` in `run_benchmark_suite.py`
- **How to benchmark just one optimizer?** Not yet supported - modify code or use tier system
- **What about multi-objective optimizers?** Separate test suite in `opt/test/`

---

**Last updated**: January 1, 2026
**Related**: [benchmarks/README.md](README.md), [PR #128](https://github.com/Anselmoo/useful-optimizer/pull/128)
