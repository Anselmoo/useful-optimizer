/**
 * Pure transforms from the published benchmark cells to chart series and table rows.
 * Everything here works on precomputed data: no run-level processing in the browser.
 */
import type { Cell, OptimizerCell } from '../types/benchmark'

/** Convergence curve of one optimizer: median with its interquartile band. */
export interface ConvergenceSeries {
  algorithm: string
  /** x values: function evaluations per dimension. */
  iterations: number[]
  /** Median precision. */
  mean: number[]
  lower: number[]
  upper: number[]
}

export interface ViolinSeries {
  algorithm: string
  values: number[]
}

export interface ECDFSeries {
  algorithm: string
  budget: number[]
  proportion: number[]
}

export interface LineSeries {
  algorithm: string
  x: number[]
  y: number[]
}

export type ScalingMetric = 'median' | 'ert' | 'time'

export interface StatsRow {
  name: string
  runs: number
  runsPlanned: number
  truncated: number
  median: number | null
  q1: number | null
  q3: number | null
  best: number | null
  worst: number | null
  success: number | null
  ert: number | null
  sp1: number | null
  microsecondsPerEval: number | null
  p: number | null
  pHolm: number | null
  delta: number | null
}

export interface Summary {
  n: number
  mean: number
  std: number
  min: number
  q1: number
  median: number
  q3: number
  max: number
}

/** Linear-interpolated quantile of an ascending-sorted array (type 7). */
export const quantileSorted = (sorted: number[], p: number) => {
  if (sorted.length === 0) return Number.NaN
  const position = (sorted.length - 1) * p
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)
}

export const summarize = (values: number[]): Summary | null => {
  const clean = values.filter(v => typeof v === 'number' && Number.isFinite(v))
  if (clean.length === 0) return null
  const sorted = [...clean].sort((a, b) => a - b)
  const n = sorted.length
  const mean = sorted.reduce((sum, v) => sum + v, 0) / n
  const variance = sorted.reduce((sum, v) => sum + (v - mean) ** 2, 0) / n
  return {
    n,
    mean,
    std: Math.sqrt(variance),
    min: sorted[0],
    q1: quantileSorted(sorted, 0.25),
    median: quantileSorted(sorted, 0.5),
    q3: quantileSorted(sorted, 0.75),
    max: sorted[n - 1]
  }
}

/** Compare against the baseline at this alpha. */
export const ALPHA = 0.05

const present = (cell: Cell, names: string[]) =>
  names.filter(name => cell.optimizers[name])

const evaluationsPerDim = (cell: Cell) => cell.checkpoints.map(c => c / cell.dim)

export const convergenceSeries = (cell: Cell, names: string[]): ConvergenceSeries[] => {
  const x = evaluationsPerDim(cell)
  const series: ConvergenceSeries[] = []
  for (const name of present(cell, names)) {
    const curve = cell.optimizers[name].convergence
    if (!curve) continue
    const keep = x
      .map((_, i) => i)
      .filter(i => curve.median[i] !== null && curve.q1[i] !== null && curve.q3[i] !== null)
    if (keep.length === 0) continue
    series.push({
      algorithm: name,
      iterations: keep.map(i => x[i]),
      mean: keep.map(i => curve.median[i] as number),
      lower: keep.map(i => curve.q1[i] as number),
      upper: keep.map(i => curve.q3[i] as number)
    })
  }
  return series
}

export const violinSeries = (cell: Cell, names: string[]): ViolinSeries[] =>
  present(cell, names)
    .map(name => ({ algorithm: name, values: cell.optimizers[name].finals }))
    .filter(series => series.values.length > 0)

export const ecdfSeries = (cell: Cell, names: string[]): ECDFSeries[] => {
  const series: ECDFSeries[] = []
  for (const name of present(cell, names)) {
    const ecdf = cell.optimizers[name].ecdf
    if (ecdf) series.push({ algorithm: name, budget: cell.ecdf_budgets, proportion: ecdf })
  }
  return series
}

export const dispersionSeries = (cell: Cell, names: string[]): LineSeries[] => {
  const x = evaluationsPerDim(cell)
  const series: LineSeries[] = []
  for (const name of present(cell, names)) {
    const values = cell.optimizers[name].dispersion
    if (!values) continue
    const keep = x.map((_, i) => i).filter(i => values[i] !== null)
    if (keep.length > 0) {
      series.push({ algorithm: name, x: keep.map(i => x[i]), y: keep.map(i => values[i] as number) })
    }
  }
  return series
}

const metricOf = (data: OptimizerCell, metric: ScalingMetric, targetIndex: number) => {
  if (metric === 'median') return data.stats?.median ?? null
  if (metric === 'ert') return data.ert[targetIndex] ?? null
  return data.time?.median_us_per_evaluation ?? null
}

/** One value per dimension for every optimizer, from the cells of that function. */
export const scalingSeries = (
  cells: Record<number, Cell>,
  names: string[],
  metric: ScalingMetric,
  targetIndex: number
): LineSeries[] => {
  const dims = Object.keys(cells).map(Number).sort((a, b) => a - b)
  const series: LineSeries[] = []
  for (const name of names) {
    const points = dims
      .map(dim => ({ dim, value: cells[dim].optimizers[name] ? metricOf(cells[dim].optimizers[name], metric, targetIndex) : null }))
      .filter((p): p is { dim: number; value: number } => p.value !== null)
    if (points.length > 0) {
      series.push({ algorithm: name, x: points.map(p => p.dim), y: points.map(p => p.value) })
    }
  }
  return series
}

export const statsRows = (cell: Cell, names: string[], targetIndex: number): StatsRow[] =>
  present(cell, names).map(name => {
    const d = cell.optimizers[name]
    return {
      name,
      runs: d.runs,
      runsPlanned: d.runs_planned,
      truncated: d.truncated,
      median: d.stats?.median ?? null,
      q1: d.stats?.q1 ?? null,
      q3: d.stats?.q3 ?? null,
      best: d.stats?.min ?? null,
      worst: d.stats?.max ?? null,
      success: d.success[targetIndex] ?? null,
      ert: d.ert[targetIndex] ?? null,
      sp1: d.sp1[targetIndex] ?? null,
      microsecondsPerEval: d.time?.median_us_per_evaluation ?? null,
      p: d.test?.p ?? null,
      pHolm: d.test?.p_holm ?? null,
      delta: d.test?.delta ?? null
    }
  })

/** `***` p < 0.001, `**` p < 0.01, `*` p < 0.05 (Holm-adjusted), empty otherwise. */
export const significanceMarker = (pHolm: number | null) => {
  if (pHolm === null) return ''
  if (pHolm < 0.001) return '***'
  if (pHolm < 0.01) return '**'
  return pHolm < ALPHA ? '*' : ''
}

export const formatNumber = (value: number | null, digits = 3) => {
  if (value === null || Number.isNaN(value)) return '–'
  if (value === 0) return '0'
  const abs = Math.abs(value)
  return abs >= 1e4 || abs < 1e-3 ? value.toExponential(digits - 1) : String(Number(value.toPrecision(digits)))
}

export const toCsv = (header: string[], rows: (string | number | null)[][]) =>
  [header, ...rows]
    .map(row => row.map(cell => (cell === null ? '' : String(cell).includes(',') ? `"${cell}"` : cell)).join(','))
    .join('\n')

/** Trigger a client-side download of `text`. */
export const downloadText = (filename: string, text: string, type = 'text/csv') => {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
