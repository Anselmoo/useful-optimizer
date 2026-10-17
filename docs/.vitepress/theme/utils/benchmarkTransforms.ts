import type { BenchmarkDataSchema } from '../types/benchmark'

type Run = BenchmarkDataSchema['benchmarks'][string][string][string]['runs'][number]

/** Convergence curve of one optimizer, aggregated over its runs. */
export interface ConvergenceSeries {
  algorithm: string
  /** x values: iterations, or function evaluations when requested. */
  iterations: number[]
  /** Central curve: the median (default) or the mean. */
  mean: number[]
  /** Only for the mean statistic. */
  std?: number[]
  /** Lower / upper edge of the band (interquartile range for the median). */
  lower?: number[]
  upper?: number[]
}

export interface ViolinSeries {
  algorithm: string
  values: number[]
}

export interface ECDFSeries {
  algorithm: string
  /** Function evaluations / dimension, strictly positive and increasing. */
  budget: number[]
  /** Fraction of (run, target) pairs reached; a right-continuous step function. */
  proportion: number[]
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

/** Default ECDF targets (absolute precision |f - f*|), COCO-style decades. */
export const DEFAULT_TARGETS = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8]

const toDimKey = (dim: string | number) =>
  typeof dim === 'string' ? dim : String(dim)

const isNum = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const filterAlgorithms = (
  optimizers: Record<string, { runs: Run[] }>,
  algorithms?: string[]
) => {
  if (!algorithms || algorithms.length === 0) return optimizers
  return Object.fromEntries(
    Object.entries(optimizers).filter(([name]) => algorithms.includes(name))
  )
}

const getOptimizerRuns = (
  data: BenchmarkDataSchema | null,
  funcName: string,
  dim: string | number,
  algorithms?: string[]
): Record<string, Run[]> => {
  if (!data?.benchmarks?.[funcName]) return {}
  const dimKey = toDimKey(dim)
  const funcEntry = data.benchmarks[funcName]
  if (!funcEntry?.[dimKey]) return {}

  const optimizers = filterAlgorithms(funcEntry[dimKey], algorithms)
  return Object.fromEntries(
    Object.entries(optimizers).map(([name, details]) => [
      name,
      details.runs ?? []
    ])
  )
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/** Linear-interpolated quantile of an ascending-sorted array (type 7). */
export const quantileSorted = (sorted: number[], p: number) => {
  if (sorted.length === 0) return Number.NaN
  const position = (sorted.length - 1) * p
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)
}

export const summarize = (values: number[]): Summary | null => {
  const clean = values.filter(isNum)
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

// ---------------------------------------------------------------------------
// Run traces
// ---------------------------------------------------------------------------

/** Best-so-far trace of one run with its real position on the iteration axis. */
export interface RunTrace {
  /** 0-based iteration of every kept point, ascending. */
  iterations: number[]
  /** Best fitness so far at each point. */
  values: number[]
  evalsPerIteration: number
  totalEvaluations: number
  /** False when only the final fitness is known. */
  hasHistory: boolean
}

/**
 * Turn a published run into a trace.
 *
 * Data written by the current aggregator carries `history.iterations`. Older
 * files only hold a 100-point downsample, so the positions are reconstructed
 * from `maxIterations` (the suite budget) when the length matches.
 */
export const toTrace = (run: Run, maxIterations?: number): RunTrace => {
  const totalEvaluations = isNum(run.n_evaluations) ? run.n_evaluations : 0
  const history = (run.history?.best_fitness ?? []).filter(isNum)

  if (history.length > 0) {
    let iterations = run.history?.iterations
    let nIterations = run.history?.n_iterations

    if (!iterations || iterations.length !== run.history?.best_fitness?.length) {
      const legacyDownsample =
        history.length >= 100 && isNum(maxIterations) && maxIterations + 1 > history.length
      nIterations = legacyDownsample ? maxIterations! + 1 : history.length
      const last = history.length - 1 || 1
      iterations = history.map((_, i) =>
        legacyDownsample ? Math.round((i * (nIterations! - 1)) / last) : i
      )
    }

    const length = nIterations ?? history.length
    return {
      iterations,
      values: history,
      evalsPerIteration: length > 0 ? totalEvaluations / length : 0,
      totalEvaluations,
      hasHistory: true
    }
  }

  return {
    iterations: [0],
    values: isNum(run.best_fitness) ? [run.best_fitness] : [],
    evalsPerIteration: totalEvaluations,
    totalEvaluations,
    hasHistory: false
  }
}

/** Index of the last element <= target (or 0 when target precedes the array). */
const lastIndexAtOrBefore = (sorted: number[], target: number) => {
  let lo = 0
  let hi = sorted.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (sorted[mid] <= target) lo = mid
    else hi = mid - 1
  }
  return lo
}

export interface ConvergenceOptions {
  /** Central curve; `median` uses the interquartile range as band. */
  statistic?: 'median' | 'mean'
  /** Show the iteration or the number of function evaluations on x. */
  xUnit?: 'iteration' | 'evaluation'
  /** Suite budget, needed to place points in legacy data. */
  maxIterations?: number
  /** Upper limit of x positions per curve. */
  maxPoints?: number
}

const thin = (values: number[], limit: number) => {
  if (values.length <= limit) return values
  const out: number[] = []
  for (let i = 0; i < limit; i += 1) {
    out.push(values[Math.round((i * (values.length - 1)) / (limit - 1))])
  }
  return Array.from(new Set(out))
}

export const buildConvergenceSeries = (
  data: BenchmarkDataSchema | null,
  funcName: string,
  dim: string | number,
  algorithms?: string[],
  options: ConvergenceOptions = {}
): ConvergenceSeries[] => {
  const { statistic = 'median', xUnit = 'iteration', maxIterations, maxPoints = 300 } = options
  const runsByOptimizer = getOptimizerRuns(data, funcName, dim, algorithms)
  const series: ConvergenceSeries[] = []

  for (const [optimizer, runs] of Object.entries(runsByOptimizer)) {
    const traces = runs.map(run => toTrace(run, maxIterations)).filter(t => t.hasHistory)
    if (traces.length === 0) continue

    // Common grid: every iteration at which at least one run has a point.
    const grid = thin(
      Array.from(new Set(traces.flatMap(t => t.iterations))).sort((a, b) => a - b),
      maxPoints
    )
    const evalsPerIteration =
      traces.reduce((sum, t) => sum + t.evalsPerIteration, 0) / traces.length

    const x: number[] = []
    const central: number[] = []
    const lower: number[] = []
    const upper: number[] = []
    const std: number[] = []

    for (const iteration of grid) {
      // Best-so-far is a step function: hold the last recorded value.
      const values = traces.map(t => t.values[lastIndexAtOrBefore(t.iterations, iteration)])
      const stats = summarize(values)
      if (!stats) continue
      x.push(xUnit === 'evaluation' ? (iteration + 1) * evalsPerIteration : iteration)
      if (statistic === 'median') {
        central.push(stats.median)
        lower.push(stats.q1)
        upper.push(stats.q3)
      } else {
        central.push(stats.mean)
        lower.push(stats.mean - stats.std)
        upper.push(stats.mean + stats.std)
        std.push(stats.std)
      }
    }

    series.push({
      algorithm: optimizer,
      iterations: x,
      mean: central,
      std: statistic === 'mean' && std.some(v => v > 0) ? std : undefined,
      lower,
      upper
    })
  }
  return series
}

export const buildViolinSeries = (
  data: BenchmarkDataSchema | null,
  funcName: string,
  dim: string | number,
  algorithms?: string[]
): ViolinSeries[] => {
  const runsByOptimizer = getOptimizerRuns(data, funcName, dim, algorithms)
  return Object.entries(runsByOptimizer).map(([optimizer, runs]) => ({
    algorithm: optimizer,
    values: runs.map(run => run.best_fitness).filter(isNum)
  }))
}

/**
 * Empirical cumulative distribution of the budget needed to reach each target
 * (COCO "runtime ECDF"): for every (run, target) pair the first evaluation count
 * at which the best fitness is at or below the target; unreached pairs never
 * count. Runs without history only contribute their final result.
 */
export const buildECDFSeries = (
  data: BenchmarkDataSchema | null,
  funcName: string,
  dim: string | number,
  targets: number[] = DEFAULT_TARGETS,
  algorithms?: string[],
  maxIterations?: number
): ECDFSeries[] => {
  const dimValue = Math.max(1, typeof dim === 'string' ? Number(dim) : dim)
  const runsByOptimizer = getOptimizerRuns(data, funcName, dim, algorithms)
  const series: ECDFSeries[] = []

  for (const [optimizer, runs] of Object.entries(runsByOptimizer)) {
    if (runs.length === 0 || targets.length === 0) continue
    const traces = runs.map(run => toTrace(run, maxIterations))
    const total = traces.length * targets.length

    const hits: number[] = []
    for (const trace of traces) {
      for (const target of targets) {
        const k = trace.values.findIndex(v => v <= target)
        if (k >= 0) {
          const evaluations = (trace.iterations[k] + 1) * trace.evalsPerIteration
          if (evaluations > 0) hits.push(evaluations / dimValue)
        }
      }
    }
    hits.sort((a, b) => a - b)

    const maxBudget = Math.max(
      ...traces.map(t => t.totalEvaluations / dimValue),
      hits[hits.length - 1] ?? 1
    )
    const start = Math.min(1, hits[0] ?? 1)

    const points = new Map<number, number>([[start, 0]])
    hits.forEach((budget, i) => points.set(budget, (i + 1) / total))
    points.set(maxBudget, hits.length / total)

    const budget = [...points.keys()].sort((a, b) => a - b)
    series.push({
      algorithm: optimizer,
      budget,
      proportion: budget.map(b => points.get(b) ?? 0)
    })
  }
  return series
}

export const fetchBenchmarkData = async (
  path: string
): Promise<BenchmarkDataSchema> => {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`Failed to load benchmark data from ${path}: ${response.status} ${response.statusText}`)
  }
  const contentType = response.headers.get('content-type') ?? ''
  // The dev server answers a missing file with index.html and status 200.
  if (contentType.includes('text/html')) {
    throw new Error(`Failed to load benchmark data from ${path}: got HTML instead of JSON`)
  }
  const json = (await response.json()) as BenchmarkDataSchema
  if (!json || typeof json !== 'object' || !json.benchmarks) {
    throw new Error(`Benchmark data at ${path} has no "benchmarks" section`)
  }
  return json
}
