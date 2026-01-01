# Benchmark Test Data

This directory contains mock benchmark data for testing the visualization components.

## Files

- **`mock-benchmark-data.json`** - Mock benchmark results following the `benchmark-data-schema.json` specification

## Data Structure

The mock data follows the IOHprofiler-compatible schema with:

### Metadata
- **max_iterations**: 100
- **n_runs**: 10 (though actual data has varying run counts per optimizer)
- **dimensions**: [2, 5, 10]
- **timestamp**: ISO 8601 format
- **python_version**: 3.12.3
- **numpy_version**: 1.26.4

### Benchmark Functions

The dataset includes results for:

1. **shifted_ackley**
   - Dimension 2: ParticleSwarm (3 runs), DifferentialEvolution (3 runs)
   - Dimension 5: ParticleSwarm (2 runs)

2. **rosenbrock**
   - Dimension 2: ParticleSwarm (2 runs)

3. **sphere**
   - Dimension 2: ParticleSwarm (1 run), DifferentialEvolution (1 run)

### Data per Optimizer

Each optimizer configuration includes:
- **runs**: Array of individual optimization runs
  - `best_fitness`: Final best fitness value
  - `best_solution`: Best solution vector
  - `n_evaluations`: Number of function evaluations
  - `history`: Convergence history (best_fitness and mean_fitness arrays)

- **statistics**: Aggregate statistics across runs
  - `mean_fitness`, `std_fitness`
  - `min_fitness`, `max_fitness`, `median_fitness`
  - `q1_fitness`, `q3_fitness` (quartiles)

- **success_rate**: Proportion of successful runs (0.0 to 1.0)

## Validation

Validate the data against the schema using the test script:

```bash
./scripts/test-benchmark-data.sh
```

This will:
1. ✓ Check JSON syntax
2. ✓ Validate against `benchmark-data-schema.json`
3. ✓ Display data summary
4. ✓ Test TypeScript type compatibility (if available)

## Usage in VitePress

### Import the Data

```vue
<script setup>
import mockData from '/test-data/mock-benchmark-data.json'
</script>
```

### Transform for Components

#### ConvergenceChart

```vue
<script setup>
import { computed } from 'vue'
import mockData from '/test-data/mock-benchmark-data.json'

const convergenceData = computed(() => {
  const ackleyData = mockData.benchmarks.shifted_ackley['2']
  return Object.keys(ackleyData).map(opt => {
    const firstRun = ackleyData[opt].runs[0]
    return {
      algorithm: opt,
      iterations: Array.from({length: 10}, (_, i) => i * 10),
      mean: firstRun.history.best_fitness,
      std: firstRun.history.mean_fitness.map((m, i) =>
        Math.abs(m - firstRun.history.best_fitness[i])
      )
    }
  })
})
</script>

<ClientOnly>
<ConvergenceChart :data="convergenceData" />
</ClientOnly>
```

#### ViolinPlot

```vue
<script setup>
import { computed } from 'vue'
import mockData from '/test-data/mock-benchmark-data.json'

const violinData = computed(() => {
  const ackleyData = mockData.benchmarks.shifted_ackley['2']
  return Object.keys(ackleyData).map(opt => ({
    algorithm: opt,
    values: ackleyData[opt].runs.map(run => run.best_fitness)
  }))
})
</script>

<ClientOnly>
<ViolinPlot :data="violinData" />
</ClientOnly>
```

#### ECDFChart

```vue
<script setup>
import { computed } from 'vue'
import mockData from '/test-data/mock-benchmark-data.json'

const ecdfData = computed(() => {
  const ackleyData = mockData.benchmarks.shifted_ackley['2']
  return Object.keys(ackleyData).map(opt => {
    // Custom ECDF calculation based on your needs
    const budgets = [100, 500, 1000, 1500, 2000]
    const proportions = budgets.map(budget => {
      // Your proportion calculation logic
      return 0.5 // Example
    })

    return {
      algorithm: opt,
      budget: budgets,
      proportion: proportions
    }
  })
})
</script>

<ClientOnly>
<ECDFChart :data="ecdfData" />
</ClientOnly>
```

## Complete Example

See [`/benchmark-demo.md`](/benchmark-demo) for a complete working example using all components with this mock data.

## Schema Reference

The data structure is defined in:
- **JSON Schema**: `docs/schemas/benchmark-data-schema.json`
- **TypeScript Types**: `docs/.vitepress/theme/types/benchmark.ts`
- **Pydantic Models**: `benchmarks/models.py`

## Generating Real Data

To generate real benchmark data:

```bash
# Run benchmark suite (when implemented)
python benchmarks/run_benchmark_suite.py

# Output will be saved to a similar JSON structure
```

## Notes

- All fitness values are mock data for demonstration purposes
- History arrays have 10 data points each for visualization
- Success rates vary (0.85 to 1.0) to show different scenarios
- Dimension 5 data is limited to demonstrate sparse data handling
