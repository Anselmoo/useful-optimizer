# Benchmark Visualization Demo

This page demonstrates the chart components using real mock benchmark data that validates against the schema.

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  buildConvergenceSeries,
  buildECDFSeries,
  buildViolinSeries,
  fetchBenchmarkData
} from './.vitepress/theme/utils/benchmarkTransforms'
import type { BenchmarkDataSchema } from './.vitepress/theme/types/benchmark'

const datasetPath = '/benchmarks/demo-benchmark-data.json'
const funcName = 'shifted_ackley'
const dimension = 2
const ecdfTargets = [1e-1, 1e-3, 1e-5, 1e-7]

const dataset = ref<BenchmarkDataSchema | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)

onMounted(async () => {
  try {
    dataset.value = await fetchBenchmarkData(datasetPath)
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load benchmark data'
  } finally {
    loading.value = false
  }
})

const ackleyData = computed(
  () => dataset.value?.benchmarks?.[funcName]?.[String(dimension)] ?? {}
)
const optimizers = computed(() => Object.keys(ackleyData.value))

const convergenceData = computed(() =>
  buildConvergenceSeries(dataset.value, funcName, dimension, optimizers.value)
)

const violinData = computed(() =>
  buildViolinSeries(dataset.value, funcName, dimension, optimizers.value)
)

const ecdfData = computed(() =>
  buildECDFSeries(dataset.value, funcName, dimension, ecdfTargets, optimizers.value)
)

const metadata = computed(() => dataset.value?.metadata)
const hasData = computed(() => optimizers.value.length > 0)
</script>

<div v-if="error" class="warning">
  {{ error }}
</div>
<div v-else-if="loading">
  Loading benchmark data from {{ datasetPath }}...
</div>
<div v-else-if="!hasData">
  No benchmark runs were found for {{ funcName }} ({{ dimension }}D).
</div>

<div v-else>

## Dataset Information

**Benchmark Suite Metadata:**
- **Max Iterations:** {{ metadata?.max_iterations ?? '—' }}
- **Number of Runs:** {{ metadata?.n_runs ?? '—' }}
- **Dimensions:** {{ metadata?.dimensions?.join(', ') ?? '—' }}
- **Python Version:** {{ metadata?.python_version ?? '—' }}
- **NumPy Version:** {{ metadata?.numpy_version ?? '—' }}
- **Timestamp:** {{ metadata?.timestamp ?? '—' }}

---

## Convergence Analysis

Comparison of **{{ optimizers.join(' vs ') }}** on the **Shifted Ackley** function (dimension 2):

<ClientOnly>
<ConvergenceChart
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

Statistical distribution of final fitness values across multiple runs:

<ClientOnly>
<ViolinPlot
  :data="violinData"
  title="Final Fitness Distribution: Shifted Ackley (2D)"
  yAxisLabel="Best Fitness"
  :logScale="true"
  :showBoxplot="true"
  :showPoints="true"
/>
</ClientOnly>

---

## ECDF Analysis

Empirical Cumulative Distribution Function showing the proportion of targets reached:

<ClientOnly>
<ECDFChart
  :data="ecdfData"
  title="ECDF: Shifted Ackley (2D)"
  xAxisLabel="Budget (function evaluations)"
  yAxisLabel="Proportion of targets reached"
  :logXAxis="false"
/>
</ClientOnly>

---

## 3D Fitness Landscape

Interactive 3D visualization of the Ackley function:

<ClientOnly>
<FitnessLandscape3D
  functionName="ackley"
  :xRange="[-5, 5]"
  :yRange="[-5, 5]"
  :resolution="80"
  colorScale="viridis"
  :height="500"
/>
</ClientOnly>

---

## Summary Statistics

<div class="stats-grid">
  <div v-for="opt in optimizers" :key="opt" class="stats-card">
    <h3>{{ opt }}</h3>
    <div class="stats-content">
      <div class="stat-item">
        <span class="label">Mean Fitness:</span>
        <span class="value">{{ ackleyData[opt].statistics.mean_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Std Deviation:</span>
        <span class="value">{{ ackleyData[opt].statistics.std_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Best Fitness:</span>
        <span class="value success">{{ ackleyData[opt].statistics.min_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Worst Fitness:</span>
        <span class="value">{{ ackleyData[opt].statistics.max_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Success Rate:</span>
        <span class="value">{{ (ackleyData[opt].success_rate * 100).toFixed(0) }}%</span>
      </div>
      <div class="stat-item">
        <span class="label">Number of Runs:</span>
        <span class="value">{{ ackleyData[opt].runs.length }}</span>
      </div>
    </div>
  </div>
</div>
</div>

<style scoped>
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
