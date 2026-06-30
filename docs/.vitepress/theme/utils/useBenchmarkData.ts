import { ref, computed } from 'vue'
import type { BenchmarkDataSchema, Benchmarks } from '../types/benchmark'

export function useBenchmarkData(
  functionName: string,
  dimension: number,
  optimizers?: string[]
) {
  const data = ref<BenchmarkDataSchema | null>(null)
  const loading = ref(true)
  const error = ref<string | null>(null)

  const loadData = async () => {
    try {
      loading.value = true
      error.value = null

      let response = await fetch('/benchmarks/benchmark-results.json')
      if (!response.ok) {
        response = await fetch('/benchmarks/demo-benchmark-data.json')
        if (!response.ok) {
          throw new Error('Failed to load benchmark data')
        }
      }
      data.value = await response.json()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error loading data'
    } finally {
      loading.value = false
    }
  }

  const filteredData = computed(() => {
    if (!data.value) return null
    const funcData = data.value.benchmarks[functionName]
    if (!funcData) return null
    const dimData = funcData[dimension.toString()]
    if (!dimData) return null
    if (optimizers && optimizers.length > 0) {
      const filtered: Record<string, Benchmarks> = {}
      for (const opt of optimizers) {
        if (dimData[opt]) filtered[opt] = dimData[opt] as Benchmarks
      }
      return filtered
    }
    return dimData
  })

  const metadata = computed(() => data.value?.metadata ?? null)

  const availableFunctions = computed(() =>
    data.value ? Object.keys(data.value.benchmarks) : []
  )

  const availableDimensions = computed(() => {
    if (!data.value) return []
    const funcData = data.value.benchmarks[functionName]
    return funcData ? Object.keys(funcData).map(Number) : []
  })

  const availableOptimizers = computed(() => {
    if (!data.value) return []
    const funcData = data.value.benchmarks[functionName]
    if (!funcData) return []
    const dimData = funcData[dimension.toString()]
    return dimData ? Object.keys(dimData) : []
  })

  loadData()

  return {
    data: filteredData,
    metadata,
    loading,
    error,
    availableFunctions,
    availableDimensions,
    availableOptimizers,
    reload: loadData
  }
}
