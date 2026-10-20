<script setup lang="ts">
/**
 * ViolinPlot.vue
 *
 * Distribution of final fitness values across runs: a kernel density violin
 * (Gaussian kernel, Silverman bandwidth, estimated in log10 space when the
 * axis is logarithmic) with an optional boxplot and the individual runs.
 * Supports zoom, pan and export.
 */
import { computed, ref, watch } from 'vue'
import type { EChartsOption, SeriesOption } from 'echarts'
import { catppuccinColors } from '../../themes/catppuccin'
import ChartFrame from './ChartFrame.vue'
import { summarize, quantileSorted } from '../utils/benchmarkTransforms'
import {
  LOG_FLOOR,
  colorFor,
  formatLog,
  logExtent,
  logSafe,
  titleOption,
  tooltipBase,
  useEChart,
  valueAxisStyle,
  zoomComponents
} from '../utils/useEChart'

interface ViolinData {
  algorithm: string
  values: number[] // Fitness values from multiple runs
}

interface Props {
  data: ViolinData[]
  title?: string
  yAxisLabel?: string
  logScale?: boolean
  showBoxplot?: boolean
  showPoints?: boolean
  height?: string | number
  /** One color per data entry; defaults to the shared palette. */
  colors?: string[]
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

// The toggles work without a parent listening: local state follows the props
// and every change is also emitted for v-model users.
const logScale = ref(props.logScale)
const showBoxplot = ref(props.showBoxplot)
const showPoints = ref(props.showPoints)
watch(() => props.logScale, v => (logScale.value = v))
watch(() => props.showBoxplot, v => (showBoxplot.value = v))
watch(() => props.showPoints, v => (showPoints.value = v))

const chartRef = ref<HTMLElement | null>(null)

const colorOf = (index: number) => props.colors?.[index] ?? colorFor(index)

const chartHeight = computed(() =>
  typeof props.height === 'number' ? `${props.height}px` : props.height
)

const usable = computed(() =>
  props.data
    .map((d, index) => ({
      ...d,
      index,
      values: d.values.filter(v => typeof v === 'number' && Number.isFinite(v))
    }))
    .filter(d => d.values.length > 0)
)

const GRID = 48

/** Gaussian KDE evaluated on a grid, returned in axis (data) coordinates. */
const kde = (values: number[], log: boolean) => {
  const t = values.map(v => (log ? Math.log10(logSafe(v)) : v))
  const n = t.length
  const mean = t.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(t.reduce((a, b) => a + (b - mean) ** 2, 0) / n)
  const sorted = [...t].sort((a, b) => a - b)
  const iqr = quantileSorted(sorted, 0.75) - quantileSorted(sorted, 0.25)
  const spread = Math.min(sd, iqr / 1.34) || sd
  let h = 0.9 * spread * n ** -0.2
  if (!(h > 0) || !Number.isFinite(h)) h = log ? 0.1 : Math.max(Math.abs(mean) * 0.05, 1e-9)

  const lo = sorted[0] - h
  const hi = sorted[n - 1] + h
  const ys: number[] = []
  const density: number[] = []
  for (let k = 0; k < GRID; k += 1) {
    const y = lo + ((hi - lo) * k) / (GRID - 1)
    let d = 0
    for (const ti of t) d += Math.exp(-0.5 * ((y - ti) / h) ** 2)
    ys.push(log ? 10 ** y : y)
    density.push(d)
  }
  const peak = Math.max(...density) || 1
  return { ys, density: density.map(d => d / peak) }
}

/** Deterministic pseudo-random jitter in [-0.15, 0.15] so points do not move. */
const jitter = (seed: number) => {
  const x = Math.sin(seed * 12.9898) * 43758.5453
  return (x - Math.floor(x) - 0.5) * 0.3
}

const chartOption = computed<EChartsOption>(() => {
  const log = logScale.value
  const y = (v: number) => (log ? logSafe(v) : v)
  const entries = usable.value
  const categories = entries.map(e => e.algorithm)

  const violins = entries.map(e => kde(e.values, log))
  const series: SeriesOption[] = []

  series.push({
    name: 'Density',
    type: 'custom',
    data: violins.map((v, i) => [i, v.ys[0], v.ys[GRID - 1]]),
    encode: { x: 0, y: [1, 2] },
    renderItem: (params: any, api: any) => {
      const idx = params.dataIndex as number
      const violin = violins[idx]
      const half = api.size([1, 0])[0] * 0.42
      const color = colorOf(entries[idx].index)
      const right: number[][] = []
      const left: number[][] = []
      violin.ys.forEach((yy, k) => {
        const [px, py] = api.coord([idx, yy])
        right.push([px + violin.density[k] * half, py])
        left.push([px - violin.density[k] * half, py])
      })
      return {
        type: 'polygon',
        shape: { points: [...right, ...left.reverse()] },
        style: api.style({ fill: color, opacity: 0.3, stroke: color, lineWidth: 1.5 })
      }
    },
    silent: true,
    z: 1
  })

  if (showBoxplot.value) {
    series.push({
      name: 'Boxplot',
      type: 'boxplot',
      data: entries.map(e => {
        const s = summarize(e.values)!
        return [s.min, s.q1, s.median, s.q3, s.max].map(y)
      }),
      boxWidth: [6, 16],
      itemStyle: {
        color: catppuccinColors.surface1,
        borderColor: catppuccinColors.text,
        borderWidth: 1.5
      },
      emphasis: { itemStyle: { borderColor: catppuccinColors.lavender, borderWidth: 2.5 } },
      z: 5
    })
  }

  if (showPoints.value) {
    const points: any[] = []
    entries.forEach((e, ci) => {
      e.values.forEach((value, i) => {
        points.push({
          value: [ci + jitter(ci * 1009 + i + 1), y(value)],
          itemStyle: { color: colorOf(e.index), opacity: 0.75 }
        })
      })
    })
    series.push({
      name: 'Runs',
      type: 'scatter',
      data: points,
      symbolSize: 6,
      z: 10
    })
  }

  const zoom = zoomComponents({ slider: false, zoomX: false, zoomY: true })
  const extent = log
    ? logExtent([
        ...entries.flatMap(e => e.values.map(y)),
        ...violins.flatMap(v => [v.ys[0], v.ys[GRID - 1]])
      ])
    : null

  return {
    backgroundColor: 'transparent',
    title: titleOption(props.title),
    tooltip: {
      ...tooltipBase,
      trigger: 'item',
      formatter: (params: any) => {
        if (params.seriesType === 'boxplot') {
          const [min, q1, median, q3, max] = (params.value as number[]).slice(1)
          return `<div style="font-weight:600">${categories[params.dataIndex]}</div>
            <div>Max: ${max.toExponential(2)}</div>
            <div>Q3: ${q3.toExponential(2)}</div>
            <div>Median: ${median.toExponential(2)}</div>
            <div>Q1: ${q1.toExponential(2)}</div>
            <div>Min: ${min.toExponential(2)}</div>`
        }
        if (params.seriesType === 'scatter') {
          return `${categories[Math.round(params.value[0])]}<br>Run: ${Number(params.value[1]).toExponential(3)}`
        }
        return ''
      }
    },
    legend: { show: false },
    toolbox: zoom.toolbox,
    dataZoom: zoom.dataZoom,
    grid: { left: 80, right: 24, top: 56, bottom: 72 },
    xAxis: {
      type: 'category',
      data: categories,
      axisLine: valueAxisStyle.axisLine,
      axisLabel: { color: catppuccinColors.text, rotate: categories.length > 4 ? 30 : 0, interval: 0 }
    },
    yAxis: {
      type: log ? 'log' : 'value',
      name: props.yAxisLabel,
      nameLocation: 'middle',
      nameGap: 62,
      scale: !log,
      splitNumber: log ? 8 : 5,
      min: extent?.min,
      max: extent?.max,
      ...valueAxisStyle,
      axisLabel: { ...valueAxisStyle.axisLabel, formatter: log ? formatLog : undefined }
    },
    series
  }
})

useEChart(chartRef, chartOption)

const clamped = computed(
  () => logScale.value && usable.value.some(e => e.values.some(v => !(v > LOG_FLOOR)))
)

const toggle = (key: 'logScale' | 'showBoxplot' | 'showPoints', event: Event) => {
  const checked = (event.target as HTMLInputElement).checked
  if (key === 'logScale') logScale.value = checked
  else if (key === 'showBoxplot') showBoxplot.value = checked
  else showPoints.value = checked
  if (key === 'logScale') emit('update:logScale', checked)
  else if (key === 'showBoxplot') emit('update:showBoxplot', checked)
  else emit('update:showPoints', checked)
}
</script>

<template>
  <ChartFrame>
    <template #header>
      <div class="chart-controls">
        <label class="control-item">
          <input type="checkbox" :checked="logScale" @change="toggle('logScale', $event)">
          Log scale
        </label>
        <label class="control-item">
          <input type="checkbox" :checked="showBoxplot" @change="toggle('showBoxplot', $event)">
          Show boxplot
        </label>
        <label class="control-item">
          <input type="checkbox" :checked="showPoints" @change="toggle('showPoints', $event)">
          Show runs
        </label>
      </div>
    </template>

    <div ref="chartRef" class="chart" :style="{ height: chartHeight }" />

    <template #footer>
      <p v-if="clamped" class="note">
        On the log axis, values at or below {{ LOG_FLOOR.toExponential(0) }} (including exact
        zeros) are drawn at {{ LOG_FLOOR.toExponential(0) }}.
      </p>
    </template>
  </ChartFrame>
</template>

<style scoped>
.chart-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
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

.note {
  margin: 0 0 12px;
  color: var(--ctp-mocha-overlay1, #7f849c);
  font-size: 0.75rem;
}
</style>
