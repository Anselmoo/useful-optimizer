<script setup lang="ts">
/**
 * ConvergenceChart.vue
 *
 * Convergence curves with a variability band, following COCO/IOHprofiler
 * conventions. The centre line and the band are supplied by the caller
 * (median + interquartile range by default, see benchmarkTransforms.ts).
 * Supports zoom, pan, box zoom, reset and PNG export.
 */
import { computed, ref } from 'vue'
import type { EChartsOption, SeriesOption } from 'echarts'
import ChartFrame from './ChartFrame.vue'
import {
  LOG_FLOOR,
  colorFor,
  formatLog,
  legendOption,
  logExtent,
  logSafe,
  titleOption,
  tooltipBase,
  useEChart,
  valueAxisStyle,
  zoomComponents
} from '../utils/useEChart'

interface ConvergenceData {
  algorithm: string
  iterations: number[]
  mean: number[]
  std?: number[]
  lower?: number[]
  upper?: number[]
}

interface Props {
  data: ConvergenceData[]
  title?: string
  xAxisLabel?: string
  yAxisLabel?: string
  logScale?: boolean
  showConfidenceBand?: boolean
  /** Name of the band in tooltips, e.g. "IQR" or "±σ". */
  bandLabel?: string
  height?: string | number
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Convergence Curve',
  xAxisLabel: 'Iteration',
  yAxisLabel: 'Best Fitness',
  logScale: true,
  showConfidenceBand: true,
  bandLabel: '±σ',
  height: 400
})

const chartRef = ref<HTMLElement | null>(null)

const chartHeight = computed(() =>
  typeof props.height === 'number' ? `${props.height}px` : props.height
)

/** Band edges: explicit lower/upper, else mean ± std. */
const bandOf = (algo: ConvergenceData) => {
  if (algo.lower && algo.upper) return { lower: algo.lower, upper: algo.upper }
  if (algo.std) {
    return {
      lower: algo.mean.map((m, i) => m - algo.std![i]),
      upper: algo.mean.map((m, i) => m + algo.std![i])
    }
  }
  return null
}

const clampedLog = computed(
  () =>
    props.logScale &&
    props.data.some(a => a.mean.some(v => !(v > LOG_FLOOR)))
)

const chartOption = computed<EChartsOption>(() => {
  const series: SeriesOption[] = []
  const y = (v: number) => (props.logScale ? logSafe(v) : v)

  props.data.forEach((algo, index) => {
    const color = colorFor(index)

    series.push({
      id: `main-${index}`,
      name: algo.algorithm,
      type: 'line',
      data: algo.iterations.map((x, i) => [x, y(algo.mean[i])]),
      smooth: false,
      symbol: 'none',
      lineStyle: { color, width: 2 },
      itemStyle: { color },
      z: 3
    })

    const band = props.showConfidenceBand ? bandOf(algo) : null
    if (band) {
      // One filled polygon between the lower and the upper edge. (Stacked
      // line series cannot do this: with two value axes ECharts stacks x.)
      const lower = band.lower.map(y)
      const upper = band.upper.map(y)
      series.push({
        // Same name as the curve, so the legend toggles curve and band together.
        name: algo.algorithm,
        type: 'custom',
        data: [[algo.iterations[0], algo.iterations[algo.iterations.length - 1], Math.min(...lower), Math.max(...upper)]],
        encode: { x: [0, 1], y: [2, 3] },
        renderItem: (_params: any, api: any) => {
          const top = algo.iterations.map((x, i) => api.coord([x, upper[i]]))
          const bottom = algo.iterations.map((x, i) => api.coord([x, lower[i]])).reverse()
          return {
            type: 'polygon',
            shape: { points: [...top, ...bottom] },
            style: api.style({ fill: color, opacity: 0.15, stroke: 'none' })
          }
        },
        silent: true,
        tooltip: { show: false },
        z: 1
      })
    }
  })

  const names = props.data.map(d => d.algorithm)
  const zoom = zoomComponents({ slider: true, zoomY: true })
  const extent = props.logScale
    ? logExtent(
        props.data.flatMap(a => [
          ...a.mean,
          ...(props.showConfidenceBand ? (bandOf(a)?.lower ?? []) : []),
          ...(props.showConfidenceBand ? (bandOf(a)?.upper ?? []) : [])
        ]).map(logSafe)
      )
    : null

  return {
    backgroundColor: 'transparent',
    title: titleOption(props.title),
    tooltip: {
      ...tooltipBase,
      trigger: 'axis',
      formatter: (params: any) => {
        if (!Array.isArray(params) || params.length === 0) return ''
        const x = params[0].value[0]
        let html = `<div style="font-weight:600">${props.xAxisLabel} ${Number(x).toLocaleString()}</div>`
        params
          .filter((p: any) => String(p.seriesId).startsWith('main-'))
          .forEach((p: any) => {
            html += `<div style="color:${p.color}">${p.seriesName}: ${Number(p.value[1]).toExponential(3)}</div>`
          })
        return html
      }
    },
    legend: legendOption(names),
    toolbox: zoom.toolbox,
    dataZoom: zoom.dataZoom,
    grid: { left: 70, right: 24, top: 56, bottom: 96 },
    xAxis: {
      type: 'value',
      name: props.xAxisLabel,
      nameLocation: 'middle',
      nameGap: 28,
      ...valueAxisStyle,
      axisLabel: { ...valueAxisStyle.axisLabel, hideOverlap: true }
    },
    yAxis: {
      type: props.logScale ? 'log' : 'value',
      name: props.yAxisLabel,
      nameLocation: 'middle',
      nameGap: 52,
      scale: !props.logScale,
      splitNumber: props.logScale ? 8 : 5,
      min: extent?.min,
      max: extent?.max,
      ...valueAxisStyle,
      axisLabel: {
        ...valueAxisStyle.axisLabel,
        formatter: props.logScale ? formatLog : undefined
      }
    },
    series
  }
})

useEChart(chartRef, chartOption)
</script>

<template>
  <ChartFrame>
    <div ref="chartRef" class="chart" :style="{ height: chartHeight }" />
    <template #footer>
      <p v-if="showConfidenceBand" class="note">Shaded band: {{ bandLabel }}.</p>
      <p v-if="clampedLog" class="note">
        On the log axis, values at or below {{ LOG_FLOOR.toExponential(0) }} (including exact
        zeros) are drawn at {{ LOG_FLOOR.toExponential(0) }}.
      </p>
    </template>
  </ChartFrame>
</template>

<style scoped>
.chart {
  width: 100%;
  min-height: 300px;
}

.note {
  margin: 0;
  color: var(--ctp-mocha-overlay1, #7f849c);
  font-size: 0.75rem;
}
</style>
