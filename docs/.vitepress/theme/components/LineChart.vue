<script setup lang="ts">
/**
 * LineChart.vue
 *
 * Generic multi-series line chart for the benchmark pages: scaling with
 * dimension (log-log) and dispersion of evaluated points over the budget.
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

interface LineData {
  algorithm: string
  x: number[]
  y: number[]
}

interface Props {
  data: LineData[]
  title?: string
  xAxisLabel?: string
  yAxisLabel?: string
  logX?: boolean
  logY?: boolean
  /** Mark the points (few x values, e.g. dimensions). */
  showSymbols?: boolean
  /** Explanation above the plot. */
  description?: string
  height?: string | number
  colors?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  xAxisLabel: '',
  yAxisLabel: '',
  logX: false,
  logY: false,
  showSymbols: false,
  description: '',
  height: 400
})

const chartRef = ref<HTMLElement | null>(null)

const colorOf = (index: number) => props.colors?.[index] ?? colorFor(index)

const chartHeight = computed(() =>
  typeof props.height === 'number' ? `${props.height}px` : props.height
)

const clampedLog = computed(
  () => props.logY && props.data.some(a => a.y.some(v => !(v > LOG_FLOOR)))
)

const chartOption = computed<EChartsOption>(() => {
  const y = (v: number) => (props.logY ? logSafe(v) : v)
  const series: SeriesOption[] = props.data.map((algo, index) => ({
    name: algo.algorithm,
    type: 'line',
    data: algo.x
      .map((x, i) => [x, y(algo.y[i])])
      .filter(([x]) => !props.logX || x > 0),
    smooth: false,
    symbol: props.showSymbols ? 'circle' : 'none',
    symbolSize: 7,
    lineStyle: { color: colorOf(index), width: 2 },
    itemStyle: { color: colorOf(index) }
  }))

  const names = props.data.map(d => d.algorithm)
  const zoom = zoomComponents({ slider: true, zoomY: true })
  const extent = props.logY ? logExtent(props.data.flatMap(a => a.y).map(logSafe)) : null

  return {
    backgroundColor: 'transparent',
    title: titleOption(props.title),
    tooltip: {
      ...tooltipBase,
      trigger: 'axis',
      formatter: (params: any) => {
        if (!Array.isArray(params) || params.length === 0) return ''
        let html = `<div style="font-weight:600">${props.xAxisLabel} ${formatLog(Number(params[0].value[0]))}</div>`
        params.forEach((p: any) => {
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
      type: props.logX ? 'log' : 'value',
      name: props.xAxisLabel,
      nameLocation: 'middle',
      nameGap: 28,
      scale: !props.logX,
      ...valueAxisStyle,
      axisLabel: { ...valueAxisStyle.axisLabel, formatter: formatLog, hideOverlap: true }
    },
    yAxis: {
      type: props.logY ? 'log' : 'value',
      name: props.yAxisLabel,
      nameLocation: 'middle',
      nameGap: 52,
      scale: !props.logY,
      splitNumber: props.logY ? 8 : 5,
      min: extent?.min,
      max: extent?.max,
      ...valueAxisStyle,
      axisLabel: { ...valueAxisStyle.axisLabel, formatter: props.logY ? formatLog : undefined }
    },
    series
  }
})

useEChart(chartRef, chartOption)
</script>

<template>
  <ChartFrame>
    <template v-if="description" #header>
      <p class="description">{{ description }}</p>
    </template>
    <div ref="chartRef" class="chart" :style="{ height: chartHeight }" />
    <template v-if="clampedLog" #footer>
      <p class="note">
        On the log axis, values at or below {{ LOG_FLOOR.toExponential(0) }} (including exact
        zeros) are drawn at {{ LOG_FLOOR.toExponential(0) }}.
      </p>
    </template>
  </ChartFrame>
</template>

<style scoped>
.description,
.note {
  margin: 0;
}

.note {
  color: var(--ctp-mocha-overlay1, #7f849c);
  font-size: 0.75rem;
}

.chart {
  width: 100%;
  min-height: 300px;
}
</style>
