# Real Benchmark Data Integration Demo

This page demonstrates the chart components using real benchmark data loaded from the data pipeline.

<script setup>
import { computed } from 'vue'
import { useBenchmarkData } from '../.vitepress/utils/useBenchmarkData'
import { 
  toMultiConvergenceData, 
  toMultiECDFData, 
  toMultiViolinData 
} from '../.vitepress/utils/benchmarkToECharts'

// Load benchmark data for shifted_ackley function, dimension 2
const { 
  data: benchmarkData, 
  loading, 
  error,
  metadata,
  availableOptimizers 
} = useBenchmarkData('shifted_ackley', 2)

// Transform data for charts
const convergenceData = computed(() => {
  if (!benchmarkData.value) return []
  return toMultiConvergenceData(benchmarkData.value)
})

const ecdfData = computed(() => {
  if (!benchmarkData.value) return []
  return toMultiECDFData(benchmarkData.value, [1e-1, 1e-3, 1e-5, 1e-7], 2)
})

const violinData = computed(() => {
  if (!benchmarkData.value) return []
  return toMultiViolinData(benchmarkData.value)
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
