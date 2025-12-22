<script setup lang="ts">
/**
 * ECDFChart.vue
 *
 * Empirical Cumulative Distribution Function chart - the gold standard
 * for optimizer comparison following COCO platform standards.
 */
import { ref, computed, onMounted, onUnmounted, watch, shallowRef } from 'vue'
import * as echarts from 'echarts'
import { catppuccinMochaTheme, catppuccinColors } from '../../themes/catppuccin'

interface ECDFData {
  algorithm: string
  budget: number[]      // Function evaluations / dimension
  proportion: number[]  // Proportion of (function, target) pairs solved
}

interface Props {
  data: ECDFData[]
  title?: string
  xAxisLabel?: string
  yAxisLabel?: string
  logXAxis?: boolean
  targetPrecisions?: number[]
  height?: string | number
}

const props = withDefaults(defineProps<Props>(), {
  title: 'ECDF: Empirical Cumulative Distribution',
  xAxisLabel: 'log₁₀(#f-evaluations / dimension)',
  yAxisLabel: 'Proportion of targets reached',
  logXAxis: true,
  targetPrecisions: () => [1e-1, 1e-3, 1e-5, 1e-7],
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
  const series: echarts.SeriesOption[] = props.data.map((algo, index) => ({
    name: algo.algorithm,
    type: 'line',
    data: algo.budget.map((b, i) => [
      props.logXAxis ? Math.log10(b) : b,
      algo.proportion[i]
    ]),
    smooth: false,
    symbol: 'none',
    lineStyle: {
      color: algorithmColors[index % algorithmColors.length],
      width: 2.5
    },
    itemStyle: {
      color: algorithmColors[index % algorithmColors.length]
    }
  }))

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
        const budget = params[0]?.value[0]
        const budgetDisplay = props.logXAxis
          ? `10^${budget.toFixed(1)}`
          : budget.toFixed(0)
        let html = `<div style="font-weight:600">Budget: ${budgetDisplay}</div>`
        params.forEach((p: any) => {
          const value = (p.value[1] * 100).toFixed(1)
          html += `<div style="color:${p.color}">${p.seriesName}: ${value}%</div>`
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
        color: catppuccinColors.subtext0,
        formatter: (value: number) => props.logXAxis ? value.toFixed(1) : value.toFixed(0)
      },
      splitLine: {
        lineStyle: { color: catppuccinColors.surface0 }
      }
    },
    yAxis: {
      type: 'value',
      name: props.yAxisLabel,
      nameLocation: 'center',
      nameGap: 40,
      min: 0,
      max: 1,
      nameTextStyle: {
        color: catppuccinColors.subtext0
      },
      axisLine: {
        lineStyle: { color: catppuccinColors.surface2 }
      },
      axisLabel: {
        color: catppuccinColors.subtext0,
        formatter: (value: number) => (value * 100).toFixed(0) + '%'
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
    <div class="chart-description">
      <p>
        ECDF curves show the proportion of (function, target) pairs that an algorithm
        can solve within a given budget of function evaluations.
      </p>
    </div>
    <div
      ref="chartRef"
      class="chart"
      :style="{ height: chartHeight }"
    />
    <div class="target-info">
      <span class="label">Target precisions:</span>
      <span
        v-for="(t, i) in targetPrecisions"
        :key="i"
        class="target-badge"
      >
        10<sup>{{ Math.log10(t) }}</sup>
      </span>
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

.chart-description {
  color: var(--ctp-mocha-subtext0, #a6adc8);
  font-size: 0.875rem;
  margin-bottom: 12px;
}

.chart-description p {
  margin: 0;
}

.chart {
  width: 100%;
  min-height: 300px;
}

.target-info {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  color: var(--ctp-mocha-subtext1, #bac2de);
  font-size: 0.875rem;
}

.label {
  font-weight: 500;
}

.target-badge {
  background-color: var(--ctp-mocha-surface1, #45475a);
  padding: 2px 8px;
  border-radius: 4px;
  font-family: monospace;
}
</style>
