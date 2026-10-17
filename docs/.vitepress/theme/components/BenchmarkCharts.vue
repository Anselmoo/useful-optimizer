<script setup lang="ts">
/**
 * Unified benchmark chart wrapper. Loads the published benchmark data and
 * renders convergence, distribution and ECDF charts for an algorithm
 * (optionally compared against others) on one function and dimension.
 *
 * Usage:
 *   <BenchmarkCharts algorithm="ParticleSwarm" functionName="shifted_ackley" :dimension="2" />
 */
import { ref, computed, onMounted, watch } from 'vue'
import { withBase } from 'vitepress'
import type { BenchmarkDataSchema } from '../types/benchmark'
import ConvergenceChart from './ConvergenceChart.vue'
import ECDFChart from './ECDFChart.vue'
import ViolinPlot from './ViolinPlot.vue'
import {
  DEFAULT_TARGETS,
  buildConvergenceSeries,
  buildECDFSeries,
  buildViolinSeries,
  fetchBenchmarkData
} from '../utils/benchmarkTransforms'

interface Props {
  algorithm: string
  functionName: string
  dimension?: number
  showConvergence?: boolean
  showECDF?: boolean
  showViolin?: boolean
  targetPrecisions?: number[]
  compareWith?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  dimension: 2,
  showConvergence: true,
  showECDF: true,
  showViolin: true,
  targetPrecisions: () => DEFAULT_TARGETS
})

const benchmarkData = ref<BenchmarkDataSchema | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)

const statistic = ref<'median' | 'mean'>('median')
const xUnit = ref<'iteration' | 'evaluation'>('iteration')

const algorithms = computed(() => [props.algorithm, ...(props.compareWith ?? [])])
const maxIterations = computed(() => benchmarkData.value?.metadata?.max_iterations)

const loadData = async () => {
  try {
    loading.value = true
    error.value = null

    let data: BenchmarkDataSchema
    try {
      data = await fetchBenchmarkData(withBase('/benchmarks/benchmark-results.json'))
    } catch {
      data = await fetchBenchmarkData(withBase('/benchmarks/demo-benchmark-data.json'))
    }

    const funcData = data.benchmarks[props.functionName]?.[String(props.dimension)]
    if (!funcData) {
      throw new Error(`No data for ${props.functionName} dimension ${props.dimension}`)
    }
    if (!algorithms.value.some(name => funcData[name])) {
      throw new Error(`No data for algorithm(s): ${algorithms.value.join(', ')}`)
    }
    benchmarkData.value = data
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Unknown error'
  } finally {
    loading.value = false
  }
}

const convergenceData = computed(() =>
  buildConvergenceSeries(
    benchmarkData.value,
    props.functionName,
    props.dimension,
    algorithms.value,
    {
      statistic: statistic.value,
      xUnit: xUnit.value,
      maxIterations: maxIterations.value
    }
  )
)

const withoutHistory = computed(() =>
  algorithms.value.filter(
    name =>
      benchmarkData.value?.benchmarks[props.functionName]?.[String(props.dimension)]?.[name] &&
      !convergenceData.value.some(series => series.algorithm === name)
  )
)

const violinData = computed(() =>
  buildViolinSeries(benchmarkData.value, props.functionName, props.dimension, algorithms.value)
)

const ecdfData = computed(() =>
  buildECDFSeries(
    benchmarkData.value,
    props.functionName,
    props.dimension,
    props.targetPrecisions,
    algorithms.value,
    maxIterations.value
  )
)

onMounted(loadData)
watch(() => [props.algorithm, props.functionName, props.dimension, props.compareWith], loadData)
</script>

<template>
  <div class="benchmark-charts">
    <div v-if="loading" class="loading-state">
      <div class="spinner"></div>
      <p>Loading benchmark data...</p>
    </div>

    <div v-else-if="error" class="error-state">
      <p>Failed to load benchmark data</p>
      <p class="error-message">{{ error }}</p>
    </div>

    <div v-else class="charts-container">
      <div v-if="showConvergence" class="chart-section">
        <h3>Convergence Analysis — Iteration vs Precision</h3>
        <div class="options">
          <label>
            Curve
            <select v-model="statistic">
              <option value="median">Median, interquartile band</option>
              <option value="mean">Mean, ±1 σ band</option>
            </select>
          </label>
          <label>
            x-axis
            <select v-model="xUnit">
              <option value="iteration">Iterations</option>
              <option value="evaluation">Function evaluations</option>
            </select>
          </label>
        </div>
        <ConvergenceChart
          v-if="convergenceData.length > 0"
          :data="convergenceData"
          :title="`${functionName} (${dimension}D): best fitness so far`"
          :xAxisLabel="xUnit === 'iteration' ? 'Iteration' : 'Function evaluations'"
          yAxisLabel="Precision |f − f*|"
          :bandLabel="statistic === 'median' ? 'IQR' : '±σ'"
          :logScale="true"
          :showConfidenceBand="true"
        />
        <p v-if="withoutHistory.length" class="notice">
          No convergence history in the published benchmark data for
          {{ withoutHistory.join(', ') }}; it appears after the next benchmark run.
          The distribution and ECDF charts below still use its final results.
        </p>
      </div>

      <div v-if="showViolin && violinData.length > 0" class="chart-section">
        <h3>Final Fitness Distribution</h3>
        <ViolinPlot
          :data="violinData"
          :title="`Distribution: ${functionName} (${dimension}D)`"
          yAxisLabel="Best Fitness"
          :logScale="true"
          :showBoxplot="true"
          :showPoints="true"
        />
      </div>

      <div v-if="showECDF && ecdfData.length > 0" class="chart-section">
        <h3>Performance Profile (ECDF)</h3>
        <ECDFChart
          :data="ecdfData"
          :title="`ECDF: ${functionName} (${dimension}D)`"
          xAxisLabel="#f-evaluations / dimension"
          yAxisLabel="Proportion of (run, target) pairs reached"
          :logXAxis="true"
          :targetPrecisions="targetPrecisions"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.benchmark-charts {
  margin: 20px 0;
}

.loading-state,
.error-state {
  padding: 40px 20px;
  text-align: center;
  border-radius: 8px;
  background-color: var(--ctp-mocha-surface0, #313244);
}

.loading-state {
  color: var(--ctp-mocha-text, #cdd6f4);
}

.spinner {
  width: 40px;
  height: 40px;
  margin: 0 auto 16px;
  border: 4px solid var(--ctp-mocha-surface1, #45475a);
  border-top-color: var(--ctp-mocha-mauve, #cba6f7);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-state {
  color: var(--ctp-mocha-red, #f38ba8);
  border: 1px solid var(--ctp-mocha-red, #f38ba8);
}

.error-message {
  font-family: monospace;
  font-size: 0.9rem;
  margin-top: 8px;
  opacity: 0.8;
}

.charts-container {
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.chart-section h3 {
  color: var(--ctp-mocha-mauve, #cba6f7);
  margin-bottom: 16px;
  font-size: 1.3rem;
  font-weight: 600;
}

.options {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 8px;
  color: var(--ctp-mocha-subtext1, #bac2de);
  font-size: 0.875rem;
}

.options label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.options select {
  background-color: var(--ctp-mocha-surface1, #45475a);
  border: 1px solid var(--ctp-mocha-surface2, #585b70);
  border-radius: 4px;
  color: var(--ctp-mocha-text, #cdd6f4);
  padding: 4px 8px;
  font-size: 0.875rem;
}

.notice {
  margin: 8px 0 0;
  padding: 10px 14px;
  border-left: 3px solid var(--ctp-mocha-yellow, #f9e2af);
  background-color: var(--ctp-mocha-surface0, #313244);
  color: var(--ctp-mocha-subtext1, #bac2de);
  font-size: 0.875rem;
}
</style>
