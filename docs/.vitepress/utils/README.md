# Benchmark Data Utilities

Utilities for transforming benchmark data from the Python benchmark suite into formats suitable for ECharts visualization components.

## Overview

This utilities package provides:

1. **Type Definitions** (`types.ts`) - TypeScript interfaces matching the Pydantic models from `benchmarks/models.py`
2. **Data Transformers** (`benchmarkToECharts.ts`) - Functions to convert benchmark data to chart-ready formats
3. **Data Loaders** (`useBenchmarkData.ts`) - Vue composables for fetching and managing benchmark data

## File Structure

```
docs/.vitepress/utils/
├── index.ts                  # Central exports
├── types.ts                  # TypeScript type definitions
├── benchmarkToECharts.ts     # Data transformation functions
├── useBenchmarkData.ts       # Vue composables for data loading
└── README.md                 # This file
```

## Usage

### 1. Direct Data Loading in Markdown

For VitePress markdown pages, use client-side data loading with `onMounted`:

```vue
<script setup>
import { ref, computed, onMounted } from 'vue'

const benchmarkData = ref(null)
const loading = ref(true)

onMounted(async () => {
  const response = await fetch('/benchmarks/benchmark-results.json')
  const data = await response.json()
  benchmarkData.value = data.benchmarks.shifted_ackley['2']
  loading.value = false
})

const convergenceData = computed(() => {
  if (!benchmarkData.value) return []
  // Transform data inline or use utility functions
  return Object.entries(benchmarkData.value).map(([name, bench]) => ({
    algorithm: name,
    // ... transformation logic
  }))
})
</script>

<ClientOnly>
<ConvergenceChart v-if="!loading" :data="convergenceData" />
</ClientOnly>
```

### 2. Using Transformation Utilities

The transformation utilities can be used in Vue components (but not directly in VitePress markdown due to build constraints):

```typescript
import { 
  toMultiConvergenceData,
  toMultiECDFData,
  toMultiViolinData 
} from '@/.vitepress/utils/benchmarkToECharts'

// Transform benchmark data
const convergenceData = toMultiConvergenceData(benchmarksMap)
const ecdfData = toMultiECDFData(benchmarksMap, [1e-1, 1e-3, 1e-5], dimension)
const violinData = toMultiViolinData(benchmarksMap)
```

### 3. Type Definitions

All data structures are typed according to the Python Pydantic models:

```typescript
import type { 
  BenchmarkDataSchema,
  Benchmarks,
  Run,
  Statistics 
} from '@/.vitepress/utils/types'

const data: BenchmarkDataSchema = await fetch('/benchmarks/benchmark-results.json')
```

## Data Transformation Functions

### `toConvergenceData(benchmarks, algorithmName)`

Transforms benchmark runs into convergence chart data with mean and standard deviation across runs.

**Parameters:**
- `benchmarks: Benchmarks` - Benchmark results for a single algorithm
- `algorithmName: string` - Name to display in chart

**Returns:** `ConvergenceData` with iterations, mean, and std arrays

### `toMultiConvergenceData(benchmarksMap)`

Transforms multiple algorithms into convergence chart data.

**Parameters:**
- `benchmarksMap: Record<string, Benchmarks>` - Map of algorithm names to benchmark results

**Returns:** `ConvergenceData[]` - Array for multiple algorithms

### `toECDFData(benchmarks, algorithmName, targetPrecisions, dimension)`

Calculates Empirical Cumulative Distribution Function data.

**Parameters:**
- `benchmarks: Benchmarks` - Benchmark results
- `algorithmName: string` - Algorithm name
- `targetPrecisions: number[]` - Target fitness thresholds (default: `[1e-1, 1e-3, 1e-5, 1e-7]`)
- `dimension: number` - Problem dimension (default: `2`)

**Returns:** `ECDFData` with budget and proportion arrays

### `toViolinData(benchmarks, algorithmName)`

Transforms final fitness values for violin plot visualization.

**Parameters:**
- `benchmarks: Benchmarks` - Benchmark results
- `algorithmName: string` - Algorithm name

**Returns:** `ViolinData` with fitness values array

### `calculateStats(values)`

Helper function for statistical calculations.

**Parameters:**
- `values: number[]` - Array of numerical values

**Returns:** Object with `mean`, `std`, `min`, `max`, `median`, `q1`, `q3`

## Data Loaders (For Vue Components)

### `useBenchmarkData(functionName, dimension, optimizers?)`

Vue composable for loading and filtering benchmark data.

**Parameters:**
- `functionName: string` - Benchmark function name (e.g., 'shifted_ackley')
- `dimension: number` - Problem dimension
- `optimizers?: string[]` - Optional array of optimizer names to filter

**Returns:**
```typescript
{
  data: Ref<Record<string, Benchmarks> | null>,
  metadata: Ref<BenchmarkMetadata | null>,
  loading: Ref<boolean>,
  error: Ref<string | null>,
  availableFunctions: Ref<string[]>,
  availableDimensions: Ref<number[]>,
  availableOptimizers: Ref<string[]>,
  reload: () => Promise<void>
}
```

**Note:** This composable may not work in VitePress markdown due to SSR constraints. Use direct `fetch` in `onMounted` instead.

## Data Schema

Benchmark data follows this structure:

```json
{
  "metadata": {
    "max_iterations": 1000,
    "n_runs": 15,
    "dimensions": [2, 5, 10, 20],
    "timestamp": "2024-12-24T15:45:00Z"
  },
  "benchmarks": {
    "function_name": {
      "dimension": {
        "OptimizerName": {
          "runs": [
            {
              "best_fitness": 0.001,
              "best_solution": [0.0, 0.0],
              "n_evaluations": 2000,
              "history": {
                "best_fitness": [100, 50, 10, 1, 0.1, 0.01, 0.001],
                "mean_fitness": [150, 80, 40, 15, 5, 2, 1]
              }
            }
          ],
          "statistics": {
            "mean_fitness": 0.0015,
            "std_fitness": 0.0005,
            "min_fitness": 0.001,
            "max_fitness": 0.002,
            "median_fitness": 0.0015
          },
          "success_rate": 1.0
        }
      }
    }
  }
}
```

## Examples

See:
- `/docs/benchmark-data-demo.md` - Complete example with all chart types
- `/docs/benchmark-demo.md` - Original demo with embedded mock data

## Best Practices

1. **Always use `ClientOnly`** wrapper for charts in VitePress
2. **Load data in `onMounted`** hook for SSR compatibility
3. **Handle loading and error states** in your UI
4. **Use type definitions** for better IDE support
5. **Transform data inline** in markdown files (utilities work best in `.vue` components)

## Future Improvements

- [ ] Support for streaming large datasets
- [ ] Caching strategy for frequently accessed data
- [ ] Progressive loading for multiple algorithms
- [ ] Integration with VitePress data loaders
- [ ] Support for comparing across different functions/dimensions
