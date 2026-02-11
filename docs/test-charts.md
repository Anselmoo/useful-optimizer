# Chart Components Test

This page demonstrates the ECharts and TresJS visualization components using real benchmark JSON data.

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

const optimizers = computed(() => {
  const slice = dataset.value?.benchmarks?.[funcName]?.[String(dimension)]
  return slice ? Object.keys(slice) : []
})

const convergenceData = computed(() =>
  buildConvergenceSeries(dataset.value, funcName, dimension, optimizers.value)
)

const violinData = computed(() =>
  buildViolinSeries(dataset.value, funcName, dimension, optimizers.value)
)

const ecdfData = computed(() =>
  buildECDFSeries(dataset.value, funcName, dimension, ecdfTargets, optimizers.value)
)

const ready = computed(
  () => !loading.value && !error.value && convergenceData.value.length > 0
)
</script>

## 2D Charts

<div v-if="error" class="warning">
  {{ error }}
</div>
<div v-else-if="loading">
  Loading benchmark data from {{ datasetPath }}...
</div>
<div v-else-if="!ready">
  Benchmark data loaded, but no runs were found for {{ funcName }} ({{ dimension }}D).
</div>
<div v-else>
  <h3>ECDF Chart</h3>
  <ClientOnly>
    <ECDFChart
      :data="ecdfData"
      title="ECDF: Algorithm Comparison"
      :logXAxis="false"
      :targetPrecisions="ecdfTargets"
    />
  </ClientOnly>

  <h3>Convergence Chart</h3>
  <ClientOnly>
    <ConvergenceChart
      :data="convergenceData"
      title="Convergence Comparison"
    />
  </ClientOnly>

  <h3>Violin Plot</h3>
  <ClientOnly>
    <ViolinPlot
      :data="violinData"
      title="Final Fitness Distribution"
    />
  </ClientOnly>
</div>

## 3D Visualization

### Fitness Landscape

<ClientOnly>
<FitnessLandscape3D
  functionName="ackley"
  :xRange="[-5, 5]"
  :yRange="[-5, 5]"
  :resolution="100"
  colorScale="viridis"
/>
</ClientOnly>
