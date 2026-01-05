<script setup lang="ts">
/**
 * BenchmarkCharts.vue
 * 
 * Wrapper component that loads benchmark data and displays convergence,
 * ECDF, and violin plot charts for a specific optimizer on a benchmark function.
 * 
 * Usage:
 *   <BenchmarkCharts 
 *     algorithm="ParticleSwarm" 
 *     function="shifted_ackley" 
 *     :dimension="2" 
 *   />
 */
import { ref, computed, onMounted, watch } from 'vue'
import ConvergenceChart from './ConvergenceChart.vue'
import ECDFChart from './ECDFChart.vue'
import ViolinPlot from './ViolinPlot.vue'

interface Props {
  /** Name of the optimization algorithm */
  algorithm: string
  /** Benchmark function name (e.g., 'shifted_ackley', 'rosenbrock') */
  functionName: string
  /** Problem dimension */
  dimension?: number
  /** Show convergence chart */
  showConvergence?: boolean
  /** Show ECDF chart */
  showECDF?: boolean
  /** Show violin plot */
  showViolin?: boolean
  /** Target precisions for ECDF */
  targetPrecisions?: number[]
  /** Compare with other algorithms */
  compareWith?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  dimension: 2,
  showConvergence: true,
  showECDF: true,
  showViolin: true,
  targetPrecisions: () => [1e-1, 1e-3, 1e-5, 1e-7]
})

// State
const benchmarkData = ref<any>(null)
const loading = ref(true)
const error = ref<string | null>(null)

// Load benchmark data
const loadData = async () => {
  try {
    loading.value = true
    error.value = null

    const response = await fetch('/benchmarks/benchmark-results.json')
    if (!response.ok) {
      throw new Error('Failed to load benchmark data')
    }
    
    const data = await response.json()
    
    // Extract data for the specified function and dimension
    const funcData = data.benchmarks[props.functionName]?.[props.dimension.toString()]
    if (!funcData) {
      throw new Error(`No data found for ${props.functionName} dimension ${props.dimension}`)
    }
    
    // Extract data for requested algorithms
    const algorithms = [props.algorithm, ...(props.compareWith || [])]
    const filtered: Record<string, any> = {}
    
    for (const alg of algorithms) {
      if (funcData[alg]) {
        filtered[alg] = funcData[alg]
      }
    }
    
    if (Object.keys(filtered).length === 0) {
      throw new Error(`No data found for algorithm(s): ${algorithms.join(', ')}`)
    }
    
    benchmarkData.value = filtered
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Unknown error'
    console.error('Error loading benchmark data:', e)
  } finally {
    loading.value = false
  }
}

// Transform data for ConvergenceChart
const convergenceData = computed(() => {
  if (!benchmarkData.value) return []
  
  return Object.entries(benchmarkData.value).map(([name, bench]: [string, any]) => {
    const runs = bench.runs.filter((run: any) => run.history?.best_fitness)
    if (runs.length === 0) return null
    
    const allHistories = runs.map((run: any) => run.history.best_fitness)
    const maxLength = Math.max(...allHistories.map((h: number[]) => h.length))
    
    const iterations: number[] = []
    const mean: number[] = []
    const std: number[] = []
    
    for (let i = 0; i < maxLength; i++) {
      const values = allHistories.filter((h: number[]) => i < h.length).map((h: number[]) => h[i])
      if (values.length > 0) {
        iterations.push(i)
        const meanVal = values.reduce((a: number, b: number) => a + b, 0) / values.length
        mean.push(meanVal)
        
        if (values.length > 1) {
          const variance = values.map((v: number) => Math.pow(v - meanVal, 2))
            .reduce((a: number, b: number) => a + b, 0) / values.length
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
  return Object.entries(benchmarkData.value).map(([name, bench]: [string, any]) => ({
    algorithm: name,
    values: bench.runs.map((run: any) => run.best_fitness)
  }))
})

// Transform data for ECDF
const ecdfData = computed(() => {
  if (!benchmarkData.value) return []
  
  return Object.entries(benchmarkData.value).map(([name, bench]: [string, any]) => {
    const runs = bench.runs.filter((run: any) => run.history?.best_fitness)
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
        const runsReachingTarget = runs.filter((run: any) => {
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
      proportion.push(totalTargetsReached / props.targetPrecisions.length)
    }
    
    return { algorithm: name, budget, proportion }
  }).filter(Boolean)
})

// Load data on mount and when props change
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
      <p>❌ Error loading benchmark data</p>
      <p class="error-message">{{ error }}</p>
    </div>

    <div v-else class="charts-container">
      <!-- Convergence Chart -->
      <div v-if="showConvergence && convergenceData.length > 0" class="chart-section">
        <h3>Convergence Analysis</h3>
        <ConvergenceChart
          :data="convergenceData"
          :title="`${algorithm} on ${functionName} (${dimension}D)`"
          xAxisLabel="Iteration"
          yAxisLabel="Best Fitness"
          :logScale="true"
          :showConfidenceBand="true"
        />
      </div>

      <!-- Violin Plot -->
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

      <!-- ECDF Chart -->
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
