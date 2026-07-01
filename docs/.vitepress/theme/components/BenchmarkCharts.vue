<script setup lang="ts">
/**
 * Unified benchmark chart wrapper. Loads data and renders convergence,
 * violin, and ECDF charts for a given algorithm/function/dimension.
 *
 * Usage:
 *   <BenchmarkCharts algorithm="ParticleSwarm" functionName="shifted_ackley" :dimension="2" />
 */
import { ref, computed, onMounted, watch } from 'vue'
import ConvergenceChart from './ConvergenceChart.vue'
import ECDFChart from './ECDFChart.vue'
import ViolinPlot from './ViolinPlot.vue'

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
  targetPrecisions: () => [1e-1, 1e-3, 1e-5, 1e-7]
})

const benchmarkData = ref<any>(null)
const loading = ref(true)
const error = ref<string | null>(null)

const loadData = async () => {
  try {
    loading.value = true
    error.value = null

    let response = await fetch('/benchmarks/benchmark-results.json')
    if (!response.ok) {
      response = await fetch('/benchmarks/demo-benchmark-data.json')
      if (!response.ok) throw new Error('Failed to load benchmark data')
    }

    const data = await response.json()
    const funcData = data.benchmarks[props.functionName]?.[props.dimension.toString()]
    if (!funcData) {
      throw new Error(`No data for ${props.functionName} dimension ${props.dimension}`)
    }

    const algorithms = [props.algorithm, ...(props.compareWith ?? [])]
    const filtered: Record<string, any> = {}
    for (const alg of algorithms) {
      if (funcData[alg]) filtered[alg] = funcData[alg]
    }
    if (Object.keys(filtered).length === 0) {
      throw new Error(`No data for algorithm(s): ${algorithms.join(', ')}`)
    }
    benchmarkData.value = filtered
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Unknown error'
  } finally {
    loading.value = false
  }
}

const convergenceData = computed(() => {
  if (!benchmarkData.value) return []
  return Object.entries(benchmarkData.value).map(([name, bench]: [string, any]) => {
    const runs = bench.runs.filter((run: any) => run.history?.best_fitness?.length)
    if (runs.length === 0) return null
    const allHistories: number[][] = runs.map((run: any) => run.history.best_fitness)
    const maxLength = Math.max(...allHistories.map(h => h.length))
    const iterations: number[] = []
    const meanArr: number[] = []
    const stdArr: number[] = []
    for (let i = 0; i < maxLength; i++) {
      const values = allHistories.filter(h => i < h.length).map(h => h[i])
      if (values.length === 0) continue
      const mu = values.reduce((a, b) => a + b, 0) / values.length
      iterations.push(i)
      meanArr.push(mu)
      stdArr.push(
        values.length > 1
          ? Math.sqrt(values.map(v => (v - mu) ** 2).reduce((a, b) => a + b, 0) / values.length)
          : 0
      )
    }
    return { algorithm: name, iterations, mean: meanArr, std: stdArr }
  }).filter(Boolean)
})

const violinData = computed(() => {
  if (!benchmarkData.value) return []
  return Object.entries(benchmarkData.value).map(([name, bench]: [string, any]) => ({
    algorithm: name,
    values: bench.runs.map((run: any) => run.best_fitness)
  }))
})

const ecdfData = computed(() => {
  if (!benchmarkData.value) return []
  return Object.entries(benchmarkData.value).map(([name, bench]: [string, any]) => {
    const runs = bench.runs.filter((run: any) => run.history?.best_fitness?.length)
    if (runs.length === 0) return null
    const maxEvals = Math.max(...runs.map((r: any) => r.n_evaluations))
    const budgetPoints = Array.from({ length: 20 }, (_, i) =>
      Math.pow(10, Math.log10(10) + (Math.log10(maxEvals / props.dimension) - Math.log10(10)) * i / 19)
    )
    const budget: number[] = []
    const proportion: number[] = []
    for (const budgetVal of budgetPoints) {
      const absoluteBudget = budgetVal * props.dimension
      let totalTargetsReached = 0
      for (const target of props.targetPrecisions) {
        const reachedCount = runs.filter((run: any) => {
          const history: number[] = run.history.best_fitness
          const evalsPerIter = run.n_evaluations / history.length
          return history.some((val, i) => (i + 1) * evalsPerIter <= absoluteBudget && val <= target)
        }).length
        if (reachedCount > 0) totalTargetsReached++
      }
      budget.push(budgetVal)
      proportion.push(totalTargetsReached / props.targetPrecisions.length)
    }
    return { algorithm: name, budget, proportion }
  }).filter(Boolean)
})

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
      <div v-if="showConvergence && convergenceData.length > 0" class="chart-section">
        <h3>Convergence Analysis — Iteration vs Precision</h3>
        <ConvergenceChart
          :data="convergenceData"
          :title="`${functionName} (${dimension}D): iteration vs precision`"
          xAxisLabel="Iteration"
          yAxisLabel="Precision |f − f*|"
          :logScale="true"
          :showConfidenceBand="true"
        />
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
          xAxisLabel="log₁₀(#f-evaluations / dimension)"
          yAxisLabel="Proportion of targets reached"
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
</style>
