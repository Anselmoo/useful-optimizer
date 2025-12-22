<script setup lang="ts">
/**
 * ConvergenceChart.vue
 *
 * Displays convergence curves for optimization algorithms with confidence bands.
 * Follows COCO/IOHprofiler visualization standards.
 */
import { ref, computed, onMounted, onUnmounted, watch, shallowRef } from 'vue'
import * as echarts from 'echarts'
import { catppuccinMochaTheme, catppuccinColors } from '../../themes/catppuccin'

interface ConvergenceData {
  algorithm: string
  iterations: number[]
  mean: number[]
  std?: number[]
  min?: number[]
  max?: number[]
}

interface Props {
  data: ConvergenceData[]
  title?: string
  xAxisLabel?: string
  yAxisLabel?: string
  logScale?: boolean
  showConfidenceBand?: boolean
  height?: string | number
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Convergence Curve',
  xAxisLabel: 'Iteration',
  yAxisLabel: 'Best Fitness',
  logScale: true,
  showConfidenceBand: true,
  height: 400
})

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

const chartOption = computed(() => {
  const series: echarts.SeriesOption[] = []

  props.data.forEach((algo, index) => {
    const color = algorithmColors[index % algorithmColors.length]

    // Main line (mean)
    series.push({
      name: algo.algorithm,
      type: 'line',
      data: algo.iterations.map((iter, i) => [iter, algo.mean[i]]),
      smooth: false,
      symbol: 'none',
      lineStyle: {
        color: color,
        width: 2
      },
      itemStyle: {
        color: color
      }
    })

    // Confidence band (mean ± std)
    if (props.showConfidenceBand && algo.std) {
      const upperBand = algo.iterations.map((iter, i) => [iter, algo.mean[i] + algo.std![i]])
      const lowerBand = algo.iterations.map((iter, i) => [iter, algo.mean[i] - algo.std![i]])

      series.push({
        name: `${algo.algorithm} (±σ)`,
        type: 'line',
        data: [...upperBand, ...[...lowerBand].reverse()],
        smooth: false,
        symbol: 'none',
        lineStyle: { opacity: 0 },
        areaStyle: {
          color: color,
          opacity: 0.15
        },
        stack: `band-${index}`
      })
    }
  })

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
      trigger: 'axis',
      backgroundColor: catppuccinColors.mantle,
      borderColor: catppuccinColors.surface1,
      textStyle: {
        color: catppuccinColors.text
      },
      formatter: (params: any) => {
        if (!Array.isArray(params)) return ''
        const iteration = params[0]?.value[0] || 0
        let html = `<div style="font-weight:600">Iteration ${iteration}</div>`
        params.forEach((p: any) => {
          if (!p.seriesName.includes('±σ')) {
            const value = p.value[1]
            html += `<div style="color:${p.color}">${p.seriesName}: ${value.toExponential(3)}</div>`
          }
        })
        return html
      }
    },
    legend: {
      data: props.data.map(d => d.algorithm),
      bottom: 10,
      textStyle: {
        color: catppuccinColors.text
      }
    },
    grid: {
      left: '10%',
      right: '5%',
      top: '15%',
      bottom: '20%'
    },
    xAxis: {
      type: 'value',
      name: props.xAxisLabel,
      nameLocation: 'center',
      nameGap: 30,
      nameTextStyle: {
        color: catppuccinColors.subtext0
      },
      axisLine: {
        lineStyle: { color: catppuccinColors.surface2 }
      },
      axisLabel: {
        color: catppuccinColors.subtext0
      },
      splitLine: {
        lineStyle: { color: catppuccinColors.surface0 }
      }
    },
    yAxis: {
      type: props.logScale ? 'log' : 'value',
      name: props.yAxisLabel,
      nameLocation: 'center',
      nameGap: 50,
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

watch(() => [props.logScale, props.showConfidenceBand], () => {
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
    <div
      ref="chartRef"
      class="chart"
      :style="{ height: chartHeight }"
    />
  </div>
</template>

<style scoped>
.chart-container {
  background-color: var(--ctp-mocha-surface0, #313244);
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
}

.chart {
  width: 100%;
  min-height: 300px;
}
</style>
