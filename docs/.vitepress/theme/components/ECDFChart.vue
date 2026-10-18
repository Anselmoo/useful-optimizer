<script setup lang="ts">
/**
 * ECDFChart.vue
 *
 * Empirical cumulative distribution of the evaluation budget needed to reach
 * a set of target precisions (COCO runtime ECDF). Curves are right-continuous
 * step functions over a log budget axis. Supports zoom, pan and export.
 */
import { computed, ref } from 'vue'
import type { EChartsOption, SeriesOption } from 'echarts'
import ChartFrame from './ChartFrame.vue'
import {
  colorFor,
  formatLog,
  legendOption,
  titleOption,
  tooltipBase,
  useEChart,
  valueAxisStyle,
  zoomComponents
} from '../utils/useEChart'

interface ECDFData {
  algorithm: string
  budget: number[] // Function evaluations / dimension, > 0
  proportion: number[] // Proportion of (run, target) pairs reached
}

interface Props {
  data: ECDFData[]
  title?: string
  xAxisLabel?: string
  yAxisLabel?: string
  logXAxis?: boolean
  targetPrecisions?: number[]
  height?: string | number
  /** One color per data entry; defaults to the shared palette. */
  colors?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  title: 'ECDF: Empirical Cumulative Distribution',
  xAxisLabel: '#f-evaluations / dimension',
  yAxisLabel: 'Proportion of targets reached',
  logXAxis: true,
  targetPrecisions: () => [1e-1, 1e-3, 1e-5, 1e-7],
  height: 400
})

const chartRef = ref<HTMLElement | null>(null)

const colorOf = (index: number) => props.colors?.[index] ?? colorFor(index)

const chartHeight = computed(() =>
  typeof props.height === 'number' ? `${props.height}px` : props.height
)

const chartOption = computed<EChartsOption>(() => {
  const series: SeriesOption[] = props.data.map((algo, index) => ({
    name: algo.algorithm,
    type: 'line',
    // A log axis cannot hold 0 or negative budgets.
    data: algo.budget
      .map((b, i) => [b, algo.proportion[i]])
      .filter(([b]) => !props.logXAxis || b > 0),
    step: 'end',
    smooth: false,
    symbol: 'none',
    lineStyle: { color: colorOf(index), width: 2.5 },
    itemStyle: { color: colorOf(index) }
  }))

  const names = props.data.map(d => d.algorithm)
  const zoom = zoomComponents({ slider: true, zoomY: false })

  return {
    backgroundColor: 'transparent',
    title: titleOption(props.title),
    tooltip: {
      ...tooltipBase,
      trigger: 'axis',
      formatter: (params: any) => {
        if (!Array.isArray(params) || params.length === 0) return ''
        const budget = Number(params[0].value[0])
        let html = `<div style="font-weight:600">Budget: ${formatLog(budget)} evals/dim</div>`
        params.forEach((p: any) => {
          html += `<div style="color:${p.color}">${p.seriesName}: ${(p.value[1] * 100).toFixed(1)}%</div>`
        })
        return html
      }
    },
    legend: legendOption(names),
    toolbox: zoom.toolbox,
    dataZoom: zoom.dataZoom,
    grid: { left: 70, right: 24, top: 56, bottom: 96 },
    xAxis: {
      type: props.logXAxis ? 'log' : 'value',
      name: props.xAxisLabel,
      nameLocation: 'middle',
      nameGap: 28,
      ...valueAxisStyle,
      axisLabel: { ...valueAxisStyle.axisLabel, formatter: formatLog, hideOverlap: true }
    },
    yAxis: {
      type: 'value',
      name: props.yAxisLabel,
      nameLocation: 'middle',
      nameGap: 48,
      min: 0,
      max: 1,
      ...valueAxisStyle,
      axisLabel: {
        ...valueAxisStyle.axisLabel,
        formatter: (value: number) => `${(value * 100).toFixed(0)}%`
      }
    },
    series
  }
})

useEChart(chartRef, chartOption)
</script>

<template>
  <ChartFrame>
    <template #header>
      <p class="description">
        Share of (run, target) pairs an algorithm has solved within a given budget of
        function evaluations per dimension. Higher and further left is better.
      </p>
    </template>
    <div ref="chartRef" class="chart" :style="{ height: chartHeight }" />
    <template #footer>
      <div class="target-info">
        <span class="label">Targets |f − f*|:</span>
        <span v-for="(t, i) in targetPrecisions" :key="i" class="target-badge">
          10<sup>{{ Math.round(Math.log10(t)) }}</sup>
        </span>
      </div>
    </template>
  </ChartFrame>
</template>

<style scoped>
.description {
  margin: 0;
}

.chart {
  width: 100%;
  min-height: 300px;
}

.target-info {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
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
