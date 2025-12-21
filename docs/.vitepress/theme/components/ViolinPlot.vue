<script setup lang="ts">
/**
 * ViolinPlot.vue
 * 
 * Statistical distribution visualization for final fitness values
 * across multiple algorithm runs.
 */
import { ref, computed, onMounted, onUnmounted, watch, shallowRef } from 'vue'
import * as echarts from 'echarts'
import { catppuccinMochaTheme, catppuccinColors } from '../../themes/catppuccin'

interface ViolinData {
  algorithm: string
  values: number[]  // Fitness values from multiple runs
}

interface Props {
  data: ViolinData[]
  title?: string
  yAxisLabel?: string
  logScale?: boolean
  showBoxplot?: boolean
  showPoints?: boolean
  height?: string | number
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Final Fitness Distribution',
  yAxisLabel: 'Fitness',
  logScale: true,
  showBoxplot: true,
  showPoints: true,
  height: 400
})

const emit = defineEmits<{
  'update:logScale': [value: boolean]
  'update:showBoxplot': [value: boolean]
  'update:showPoints': [value: boolean]
}>()

const chartRef = ref<HTMLElement | null>(null)
const chart = shallowRef<echarts.ECharts | null>(null)

const chartHeight = computed(() => {
  return typeof props.height === 'number' ? `${props.height}px` : props.height
})

const algorithmColors = [
  catppuccinColors.mauve,
  catppuccinColors.blue,
  catppuccinColors.green,
  catppuccinColors.yellow,
  catppuccinColors.peach,
  catppuccinColors.red,
  catppuccinColors.teal,
  catppuccinColors.pink
]

// Calculate boxplot statistics
const calculateStats = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b)
  const n = sorted.length
  const q1 = sorted[Math.floor(n * 0.25)]
  const median = sorted[Math.floor(n * 0.5)]
  const q3 = sorted[Math.floor(n * 0.75)]
  const min = sorted[0]
  const max = sorted[n - 1]
  const mean = values.reduce((a, b) => a + b, 0) / n
  
  return { min, q1, median, q3, max, mean }
}

const chartOption = computed(() => {
  const categories = props.data.map(d => d.algorithm)
  
  // Boxplot data: [min, Q1, median, Q3, max]
  const boxplotData = props.data.map(d => {
    const stats = calculateStats(d.values)
    return [stats.min, stats.q1, stats.median, stats.q3, stats.max]
  })
  
  // Scatter data for individual points
  const scatterData: any[] = []
  props.data.forEach((algo, algoIndex) => {
    algo.values.forEach(value => {
      scatterData.push({
        value: [algoIndex + (Math.random() - 0.5) * 0.3, value],
        itemStyle: {
          color: algorithmColors[algoIndex % algorithmColors.length],
          opacity: 0.6
        }
      })
    })
  })

  const series: echarts.SeriesOption[] = []
  
  if (props.showBoxplot) {
    series.push({
      name: 'Boxplot',
      type: 'boxplot',
      data: boxplotData,
      itemStyle: {
        color: catppuccinColors.surface1,
        borderColor: catppuccinColors.mauve,
        borderWidth: 2
      },
      emphasis: {
        itemStyle: {
          borderColor: catppuccinColors.lavender,
          borderWidth: 3
        }
      }
    })
  }
  
  if (props.showPoints) {
    series.push({
      name: 'Data Points',
      type: 'scatter',
      data: scatterData,
      symbolSize: 6,
      z: 10
    })
  }

  return {
    backgroundColor: 'transparent',
    title: {
      text: props.title,
      left: 'center',
      textStyle: {
        color: catppuccinColors.text,
        fontSize: 16,
        fontWeight: 600
      }
    },
    tooltip: {
      trigger: 'item',
      backgroundColor: catppuccinColors.mantle,
      borderColor: catppuccinColors.surface1,
      textStyle: {
        color: catppuccinColors.text
      },
      formatter: (params: any) => {
        if (params.seriesType === 'boxplot') {
          const [min, q1, median, q3, max] = params.value
          const algo = categories[params.dataIndex]
          return `<div style="font-weight:600">${algo}</div>
            <div>Max: ${max.toExponential(2)}</div>
            <div>Q3: ${q3.toExponential(2)}</div>
            <div>Median: ${median.toExponential(2)}</div>
            <div>Q1: ${q1.toExponential(2)}</div>
            <div>Min: ${min.toExponential(2)}</div>`
        }
        return `Value: ${params.value[1].toExponential(3)}`
      }
    },
    legend: {
      show: false
    },
    grid: {
      left: '15%',
      right: '5%',
      top: '15%',
      bottom: '15%'
    },
    xAxis: {
      type: 'category',
      data: categories,
      axisLine: {
        lineStyle: { color: catppuccinColors.surface2 }
      },
      axisLabel: {
        color: catppuccinColors.text,
        rotate: 30
      }
    },
    yAxis: {
      type: props.logScale ? 'log' : 'value',
      name: props.yAxisLabel,
      nameLocation: 'center',
      nameGap: 60,
      nameTextStyle: {
        color: catppuccinColors.subtext0
      },
      axisLine: {
        lineStyle: { color: catppuccinColors.surface2 }
      },
      axisLabel: {
        color: catppuccinColors.subtext0,
        formatter: (value: number) => value.toExponential(0)
      },
      splitLine: {
        lineStyle: { color: catppuccinColors.surface0 }
      }
    },
    series
  }
})

const initChart = () => {
  if (!chartRef.value) return
  
  echarts.registerTheme('catppuccin-mocha', catppuccinMochaTheme)
  chart.value = echarts.init(chartRef.value, 'catppuccin-mocha')
  chart.value.setOption(chartOption.value)
}

const resizeChart = () => {
  chart.value?.resize()
}

watch(() => props.data, () => {
  chart.value?.setOption(chartOption.value)
}, { deep: true })

watch(() => [props.logScale, props.showBoxplot, props.showPoints], () => {
  chart.value?.setOption(chartOption.value)
})

onMounted(() => {
  initChart()
  window.addEventListener('resize', resizeChart)
})

onUnmounted(() => {
  window.removeEventListener('resize', resizeChart)
  chart.value?.dispose()
})
</script>

<template>
  <div class="chart-container">
    <div class="chart-controls">
      <label class="control-item">
        <input type="checkbox" :checked="logScale" @change="emit('update:logScale', ($event.target as HTMLInputElement).checked)">
        Log scale
      </label>
      <label class="control-item">
        <input type="checkbox" :checked="showBoxplot" @change="emit('update:showBoxplot', ($event.target as HTMLInputElement).checked)">
        Show boxplot
      </label>
      <label class="control-item">
        <input type="checkbox" :checked="showPoints" @change="emit('update:showPoints', ($event.target as HTMLInputElement).checked)">
        Show points
      </label>
    </div>
    <div 
      ref="chartRef" 
      class="chart"
      :style="{ height: chartHeight }"
    />
    <div class="chart-stats">
      <table class="stats-table">
        <thead>
          <tr>
            <th>Algorithm</th>
            <th>Mean</th>
            <th>Std</th>
            <th>Best</th>
            <th>Worst</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="algo in data" :key="algo.algorithm">
            <td>{{ algo.algorithm }}</td>
            <td>{{ (algo.values.reduce((a, b) => a + b, 0) / algo.values.length).toExponential(2) }}</td>
            <td>{{ Math.sqrt(algo.values.map(v => Math.pow(v - algo.values.reduce((a, b) => a + b, 0) / algo.values.length, 2)).reduce((a, b) => a + b, 0) / algo.values.length).toExponential(2) }}</td>
            <td class="good">{{ Math.min(...algo.values).toExponential(2) }}</td>
            <td>{{ Math.max(...algo.values).toExponential(2) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.chart-container {
  background-color: var(--ctp-mocha-surface0, #313244);
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
}

.chart-controls {
  display: flex;
  gap: 16px;
  margin-bottom: 12px;
}

.control-item {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--ctp-mocha-subtext1, #bac2de);
  font-size: 0.875rem;
  cursor: pointer;
}

.control-item input {
  accent-color: var(--ctp-mocha-mauve, #cba6f7);
}

.chart {
  width: 100%;
  min-height: 300px;
}

.chart-stats {
  margin-top: 16px;
  overflow-x: auto;
}

.stats-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.stats-table th,
.stats-table td {
  padding: 8px 12px;
  text-align: left;
  border: 1px solid var(--ctp-mocha-surface1, #45475a);
}

.stats-table th {
  background-color: var(--ctp-mocha-surface1, #45475a);
  color: var(--ctp-mocha-text, #cdd6f4);
  font-weight: 600;
}

.stats-table td {
  color: var(--ctp-mocha-subtext1, #bac2de);
  font-family: monospace;
}

.stats-table td.good {
  color: var(--ctp-mocha-green, #a6e3a1);
}
</style>
