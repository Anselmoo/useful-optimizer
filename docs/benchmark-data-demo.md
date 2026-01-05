# Real Benchmark Data Integration Demo

This page demonstrates the chart components using real benchmark data loaded from the data pipeline.

<script setup>
import { computed, ref, onMounted } from 'vue'

// State
const benchmarkData = ref(null)
const loading = ref(true)
const error = ref(null)
const metadata = ref(null)

// Load benchmark data on mount (client-side only)
onMounted(async () => {
  try {
    const response = await fetch('/benchmarks/benchmark-results.json')
    if (!response.ok) {
      throw new Error('Failed to load benchmark data')
    }
    const data = await response.json()
    
    // Extract data for shifted_ackley, dimension 2
    const funcData = data.benchmarks.shifted_ackley?.['2']
    if (funcData) {
      benchmarkData.value = funcData
      metadata.value = data.metadata
    } else {
      throw new Error('No data found for shifted_ackley dimension 2')
    }
  } catch (e) {
    error.value = e.message
    console.error('Error loading benchmark data:', e)
  } finally {
    loading.value = false
  }
})

const availableOptimizers = computed(() => {
  if (!benchmarkData.value) return []
  return Object.keys(benchmarkData.value)
})

// Transform data for ConvergenceChart
const convergenceData = computed(() => {
  if (!benchmarkData.value) return []
  
  return Object.entries(benchmarkData.value).map(([name, bench]) => {
    const runs = bench.runs.filter(run => run.history?.best_fitness)
    if (runs.length === 0) return null
    
    const allHistories = runs.map(run => run.history.best_fitness)
    const maxLength = Math.max(...allHistories.map(h => h.length))
    
    const iterations = []
    const mean = []
    const std = []
    
    for (let i = 0; i < maxLength; i++) {
      const values = allHistories.filter(h => i < h.length).map(h => h[i])
      if (values.length > 0) {
        iterations.push(i)
        const meanVal = values.reduce((a, b) => a + b, 0) / values.length
        mean.push(meanVal)
        
        if (values.length > 1) {
          const variance = values.map(v => Math.pow(v - meanVal, 2)).reduce((a, b) => a + b, 0) / values.length
          std.push(Math.sqrt(variance))
        } else {
          std.push(0)
        }
      }
    }
    
    return { algorithm: name, iterations, mean, std }
  }).filter(Boolean)
})

// Transform data for ViolinPlot
const violinData = computed(() => {
  if (!benchmarkData.value) return []
  return Object.entries(benchmarkData.value).map(([name, bench]) => ({
    algorithm: name,
    values: bench.runs.map(run => run.best_fitness)
  }))
})

// Transform data for ECDF
const ecdfData = computed(() => {
  if (!benchmarkData.value) return []
  
  const targetPrecisions = [1e-1, 1e-3, 1e-5, 1e-7]
  const dimension = 2
  
  return Object.entries(benchmarkData.value).map(([name, bench]) => {
    const runs = bench.runs.filter(run => run.history?.best_fitness)
    if (runs.length === 0) return null
    
    const maxEvals = Math.max(...runs.map(r => r.n_evaluations))
    const budgetPoints = Array.from({ length: 20 }, (_, i) => 
      Math.pow(10, Math.log10(10) + (Math.log10(maxEvals / dimension) - Math.log10(10)) * i / 19)
    )
    
    const budget = []
    const proportion = []
    
    for (const budgetVal of budgetPoints) {
      const absoluteBudget = budgetVal * dimension
      let totalTargetsReached = 0
      
      for (const target of targetPrecisions) {
        const runsReachingTarget = runs.filter(run => {
          const history = run.history.best_fitness
          const evaluationsPerIteration = run.n_evaluations / history.length
          for (let i = 0; i < history.length; i++) {
            const evals = (i + 1) * evaluationsPerIteration
            if (evals <= absoluteBudget && history[i] <= target) {
              return true
            }
          }
          return false
        })
        if (runsReachingTarget.length > 0) {
          totalTargetsReached++
        }
      }
      
      budget.push(budgetVal)
      proportion.push(totalTargetsReached / targetPrecisions.length)
    }
    
    return { algorithm: name, budget, proportion }
  }).filter(Boolean)
})

// Summary statistics
const summaryStats = computed(() => {
  if (!benchmarkData.value) return []
  return Object.entries(benchmarkData.value).map(([name, bench]) => ({
    name,
    ...bench.statistics,
    success_rate: bench.success_rate,
    n_runs: bench.runs.length
  }))
})
</script>

## Dataset Information

<div v-if="loading" class="loading">
  Loading benchmark data...
</div>

<div v-else-if="error" class="error">
  Error loading data: {{ error }}
</div>

<div v-else-if="metadata">

**Benchmark Suite Metadata:**
- **Max Iterations:** {{ metadata.max_iterations }}
- **Number of Runs:** {{ metadata.n_runs }}
- **Dimensions:** {{ metadata.dimensions.join(', ') }}
- **Python Version:** {{ metadata.python_version }}
- **NumPy Version:** {{ metadata.numpy_version }}
- **Timestamp:** {{ metadata.timestamp }}

**Available Optimizers:** {{ availableOptimizers.join(', ') }}

---

## Convergence Analysis

Real convergence data from multiple independent runs on **Shifted Ackley (2D)**:

<ClientOnly>
<ConvergenceChart
  v-if="convergenceData.length > 0"
  :data="convergenceData"
  title="Convergence: Shifted Ackley (2D)"
  xAxisLabel="Iteration"
  yAxisLabel="Best Fitness"
  :logScale="true"
  :showConfidenceBand="true"
/>
</ClientOnly>

---

## Final Fitness Distribution

Statistical distribution showing variance across multiple runs:

<ClientOnly>
<ViolinPlot
  v-if="violinData.length > 0"
  :data="violinData"
  title="Final Fitness Distribution: Shifted Ackley (2D)"
  yAxisLabel="Best Fitness"
  :logScale="true"
  :showBoxplot="true"
  :showPoints="true"
/>
</ClientOnly>

---

## ECDF: Performance Profile

Empirical Cumulative Distribution Function showing algorithm efficiency:

<ClientOnly>
<ECDFChart
  v-if="ecdfData.length > 0"
  :data="ecdfData"
  title="ECDF: Shifted Ackley (2D)"
  xAxisLabel="log₁₀(#f-evaluations / dimension)"
  yAxisLabel="Proportion of targets reached"
  :logXAxis="true"
  :targetPrecisions="[1e-1, 1e-3, 1e-5, 1e-7]"
/>
</ClientOnly>

---

## Summary Statistics

<div v-if="summaryStats.length > 0" class="stats-grid">
  <div v-for="stat in summaryStats" :key="stat.name" class="stats-card">
    <h3>{{ stat.name }}</h3>
    <div class="stats-content">
      <div class="stat-item">
        <span class="label">Mean Fitness:</span>
        <span class="value">{{ stat.mean_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Std Deviation:</span>
        <span class="value">{{ stat.std_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Best Fitness:</span>
        <span class="value success">{{ stat.min_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Worst Fitness:</span>
        <span class="value">{{ stat.max_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Median Fitness:</span>
        <span class="value">{{ stat.median_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Success Rate:</span>
        <span class="value">{{ (stat.success_rate * 100).toFixed(0) }}%</span>
      </div>
      <div class="stat-item">
        <span class="label">Number of Runs:</span>
        <span class="value">{{ stat.n_runs }}</span>
      </div>
    </div>
  </div>
</div>

</div>

<style scoped>
.loading, .error {
  padding: 20px;
  margin: 20px 0;
  border-radius: 8px;
  text-align: center;
}

.loading {
  background-color: var(--ctp-mocha-surface0, #313244);
  color: var(--ctp-mocha-text, #cdd6f4);
}

.error {
  background-color: var(--ctp-mocha-surface0, #313244);
  color: var(--ctp-mocha-red, #f38ba8);
  border: 1px solid var(--ctp-mocha-red, #f38ba8);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.stats-card {
  background-color: var(--ctp-mocha-surface0, #313244);
  border-radius: 8px;
  padding: 20px;
  border: 1px solid var(--ctp-mocha-surface1, #45475a);
}

.stats-card h3 {
  margin: 0 0 16px 0;
  color: var(--ctp-mocha-mauve, #cba6f7);
  font-size: 1.2rem;
  border-bottom: 2px solid var(--ctp-mocha-surface1, #45475a);
  padding-bottom: 8px;
}

.stats-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
}

.stat-item .label {
  color: var(--ctp-mocha-subtext0, #a6adc8);
  font-size: 0.9rem;
}

.stat-item .value {
  color: var(--ctp-mocha-text, #cdd6f4);
  font-weight: 600;
  font-family: monospace;
}

.stat-item .value.success {
  color: var(--ctp-mocha-green, #a6e3a1);
}
</style>
