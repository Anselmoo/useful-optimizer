/**
 * Shared ECharts plumbing for the benchmark chart components.
 *
 * Importing this module pulls in `echarts`, which touches `document` at load
 * time, so it must only be imported from components that are registered
 * client-side (see theme/index.ts) and rendered inside <ClientOnly>.
 */
import { onBeforeUnmount, onMounted, shallowRef, watch, type Ref } from 'vue'
import * as echarts from 'echarts'
import { catppuccinMochaTheme, catppuccinColors } from '../../themes/catppuccin'

/** Smallest value drawn on a log axis; zeros and negatives are clamped to it. */
export const LOG_FLOOR = 1e-12

export const algorithmColors = [
  catppuccinColors.mauve,
  catppuccinColors.blue,
  catppuccinColors.green,
  catppuccinColors.yellow,
  catppuccinColors.peach,
  catppuccinColors.red,
  catppuccinColors.teal,
  catppuccinColors.pink,
  catppuccinColors.sapphire,
  catppuccinColors.lavender
]

export const colorFor = (index: number) =>
  algorithmColors[index % algorithmColors.length]

/** Clamp a value so it can be drawn on a log axis. */
export const logSafe = (value: number) =>
  Number.isFinite(value) ? Math.max(value, LOG_FLOOR) : LOG_FLOOR

let themeRegistered = false

/**
 * Create an ECharts instance in `container` and keep it in sync with `option`.
 *
 * - Every update replaces the option (`notMerge`), so removed series disappear.
 * - The chart follows its container size (sidebar toggles, tabs, <details>).
 * - The instance is disposed on unmount.
 */
export function useEChart(
  container: Ref<HTMLElement | null>,
  option: Ref<echarts.EChartsOption>
) {
  const chart = shallowRef<echarts.ECharts | null>(null)
  let observer: ResizeObserver | null = null

  const render = () => {
    chart.value?.setOption(option.value, { notMerge: true })
  }

  const onWheel = (event: WheelEvent) => wheelZoom(chart.value, container.value, event)

  onMounted(() => {
    if (typeof window === 'undefined' || !container.value) return

    if (!themeRegistered) {
      echarts.registerTheme('catppuccin-mocha', catppuccinMochaTheme)
      themeRegistered = true
    }
    chart.value = echarts.init(container.value, 'catppuccin-mocha')
    render()
    // Not passive: Ctrl/⌘ + wheel must be able to cancel the browser's page zoom.
    container.value.addEventListener('wheel', onWheel, { passive: false })

    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(() => chart.value?.resize())
      observer.observe(container.value)
    }
  })

  watch(option, render)

  onBeforeUnmount(() => {
    container.value?.removeEventListener('wheel', onWheel)
    observer?.disconnect()
    observer = null
    chart.value?.dispose()
    chart.value = null
  })

  return { chart, render }
}

const axisText = { color: catppuccinColors.subtext0 }
const axisLine = { lineStyle: { color: catppuccinColors.surface2 } }
const splitLine = { lineStyle: { color: catppuccinColors.surface0 } }

/** Styling shared by every value / log axis. */
export const valueAxisStyle = {
  nameTextStyle: axisText,
  axisLine,
  axisLabel: axisText,
  splitLine
}

export const titleOption = (text: string) => ({
  text,
  left: 'center',
  textStyle: {
    color: catppuccinColors.text,
    fontSize: 16,
    fontWeight: 600
  }
})

export const tooltipBase = {
  backgroundColor: catppuccinColors.mantle,
  borderColor: catppuccinColors.surface1,
  textStyle: { color: catppuccinColors.text }
}

export const legendOption = (names: string[]) => ({
  data: names,
  bottom: 8,
  type: 'scroll' as const,
  textStyle: { color: catppuccinColors.text },
  pageTextStyle: { color: catppuccinColors.text }
})

/**
 * Decade-aligned [min, max] for a log axis. ECharts' own "nice" extent can
 * stretch far past the data (e.g. up to 1e10 for values near 50), which
 * flattens everything; this keeps the axis tight around what is plotted.
 */
export const logExtent = (values: number[]) => {
  const positive = values.filter(v => Number.isFinite(v) && v > 0)
  if (positive.length === 0) return { min: LOG_FLOOR, max: 1 }
  const lo = Math.floor(Math.log10(Math.min(...positive)))
  let hi = Math.ceil(Math.log10(Math.max(...positive)))
  if (hi <= lo) hi = lo + 1
  return { min: 10 ** lo, max: 10 ** hi }
}

/** Compact label for log axes: 1e-8 → "1e-8", 100 → "100". */
export const formatLog = (value: number) => {
  if (value === 0) return '0'
  const abs = Math.abs(value)
  if (abs >= 0.01 && abs < 10000) return String(Number(value.toPrecision(3)))
  return value.toExponential(0).replace('e+', 'e')
}


const ZOOM_IDS = ['zoom-x-inside', 'zoom-y-inside'] as const
const MIN_SPAN = 0.5 // percent of the full axis

/**
 * Ctrl/⌘ + mouse wheel (and trackpad pinch, which browsers report as Ctrl +
 * wheel) zooms around the pointer. A plain wheel is left alone so the page
 * keeps scrolling: ECharts' own wheel zoom would swallow every wheel event
 * over the plot, whatever modifier it is configured with.
 */
function wheelZoom(
  chart: echarts.ECharts | null,
  container: HTMLElement | null,
  event: WheelEvent
) {
  if (!chart || !container || !(event.ctrlKey || event.metaKey)) return

  const zooms = (chart.getOption() as any)?.dataZoom as any[] | undefined
  const grid = (chart as any).getModel?.().getComponent('grid', 0)?.coordinateSystem?.getRect?.()
  if (!zooms || !grid) return

  const bounds = container.getBoundingClientRect()
  const px = event.clientX - bounds.left
  const py = event.clientY - bounds.top
  if (px < grid.x || px > grid.x + grid.width || py < grid.y || py > grid.y + grid.height) return

  event.preventDefault()
  const factor = event.deltaY < 0 ? 0.85 : 1 / 0.85

  for (const id of ZOOM_IDS) {
    const zoom = zooms.find(z => z.id === id)
    if (!zoom) continue
    // x grows to the right, y grows upwards.
    const focus = id === 'zoom-x-inside'
      ? (px - grid.x) / grid.width
      : 1 - (py - grid.y) / grid.height
    const start = zoom.start ?? 0
    const end = zoom.end ?? 100
    const span = Math.min(100, Math.max(MIN_SPAN, (end - start) * factor))
    let nextStart = start + (end - start) * focus - span * focus
    nextStart = Math.min(100 - span, Math.max(0, nextStart))
    chart.dispatchAction({
      type: 'dataZoom',
      dataZoomId: id,
      start: nextStart,
      end: nextStart + span
    })
  }
}

interface ZoomOptions {
  /** Add a zoom slider under the plot. */
  slider?: boolean
  /** Allow zoom / pan along the y axis (numeric y axes only). */
  zoomY?: boolean
  /** Allow zoom / pan along the x axis. */
  zoomX?: boolean
}

/**
 * Zoom, pan, reset and export controls, shared by all charts.
 *
 * Ctrl/⌘ + wheel zooms (see wheelZoom), dragging pans, and the toolbox adds box
 * zoom, reset and PNG export.
 */
export function zoomComponents({
  slider = true,
  zoomY = true,
  zoomX = true
}: ZoomOptions = {}) {
  // `zoomLock` keeps the built-in roam to dragging only; wheel zoom is done by
  // wheelZoom() above so the page can still scroll over the plot.
  const inside = {
    type: 'inside' as const,
    filterMode: 'none' as const,
    zoomLock: true,
    moveOnMouseMove: true,
    preventDefaultMouseMove: false
  }

  const dataZoom: echarts.EChartsOption['dataZoom'] = []
  if (zoomX) dataZoom.push({ ...inside, id: 'zoom-x-inside', xAxisIndex: 0 })
  if (zoomY) dataZoom.push({ ...inside, id: 'zoom-y-inside', yAxisIndex: 0 })
  if (zoomX && slider) {
    dataZoom.push({
      type: 'slider',
      id: 'zoom-x-slider',
      xAxisIndex: 0,
      filterMode: 'none',
      height: 16,
      bottom: 40,
      borderColor: catppuccinColors.surface1,
      backgroundColor: catppuccinColors.mantle,
      fillerColor: 'rgba(203, 166, 247, 0.2)',
      handleStyle: { color: catppuccinColors.mauve },
      textStyle: { color: catppuccinColors.subtext0 }
    })
  }

  return {
    dataZoom,
    toolbox: {
      right: 8,
      top: 0,
      iconStyle: { borderColor: catppuccinColors.subtext0 },
      emphasis: { iconStyle: { borderColor: catppuccinColors.mauve } },
      feature: {
        dataZoom: {
          yAxisIndex: (zoomY ? 0 : 'none') as 0 | 'none',
          xAxisIndex: (zoomX ? 0 : 'none') as 0 | 'none',
          title: { zoom: 'Box zoom', back: 'Undo zoom' }
        },
        restore: { title: 'Reset view' },
        saveAsImage: {
          title: 'Save as PNG',
          backgroundColor: catppuccinColors.base,
          pixelRatio: 2
        }
      }
    }
  }
}

export const ZOOM_HINT = 'Drag to pan · Ctrl + scroll or pinch to zoom · toolbox: box zoom, reset, save PNG'
