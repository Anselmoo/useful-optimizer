import type { BenchmarkDataSchema } from '../types/benchmark'

type Run =
  BenchmarkDataSchema['benchmarks'][string][string][string]['runs'][number]

export interface ConvergenceSeries {
  algorithm: string
  iterations: number[]
  mean: number[]
  std?: number[]
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

const toDimKey = (dim: string | number) =>
  typeof dim === 'string' ? dim : String(dim)

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

const mean = (values: number[]) =>
  values.reduce((sum, value) => sum + value, 0) / (values.length || 1)

const stdDev = (values: number[], mu: number) => {
  if (values.length === 0) return 0
  const variance =
    values.reduce((sum, value) => sum + (value - mu) ** 2, 0) /
    values.length
  return Math.sqrt(variance)
}

export const buildConvergenceSeries = (
  data: BenchmarkDataSchema | null,
  funcName: string,
  dim: string | number,
  algorithms?: string[]
): ConvergenceSeries[] => {
  const runsByOptimizer = getOptimizerRuns(data, funcName, dim, algorithms)
  return Object.entries(runsByOptimizer).map(([optimizer, runs]) => {
    const histories = runs
      .map(run => run.history?.best_fitness ?? [])
      .filter(history => history.length > 0)

    const maxLength = Math.max(0, ...histories.map(history => history.length))
    const iterations: number[] = []
    const means: number[] = []
    const stds: number[] = []

    for (let i = 0; i < maxLength; i += 1) {
      const values = histories
        .map(history => history[i])
        .filter((value): value is number => typeof value === 'number')

      if (values.length === 0) continue
      const mu = mean(values)
      iterations.push(i)
      means.push(mu)
      stds.push(stdDev(values, mu))
    }

    return {
      algorithm: optimizer,
      iterations,
      mean: means,
      std: stds.some(value => value > 0) ? stds : undefined
    }
  })
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
    values: runs
      .map(run => run.best_fitness)
      .filter((value): value is number => typeof value === 'number')
  }))
}

const toNormalizedBudget = (nEvaluations: number | undefined, dim: number) => {
  if (!nEvaluations || dim <= 0) return Number.POSITIVE_INFINITY
  return nEvaluations / dim
}

export const buildECDFSeries = (
  data: BenchmarkDataSchema | null,
  funcName: string,
  dim: string | number,
  targets: number[],
  algorithms?: string[]
): ECDFSeries[] => {
  const dimValue = typeof dim === 'string' ? Number(dim) : dim
  const runsByOptimizer = getOptimizerRuns(data, funcName, dim, algorithms)

  return Object.entries(runsByOptimizer).map(([optimizer, runs]) => {
    const pairs = runs.flatMap(run =>
      targets.map(target => ({
        solved:
          typeof run.best_fitness === 'number' &&
          run.best_fitness <= target,
        budget: toNormalizedBudget(run.n_evaluations, dimValue)
      }))
    )

    const solvedBudgets = pairs
      .filter(pair => pair.solved && Number.isFinite(pair.budget))
      .map(pair => pair.budget)

    const budgetPoints = Array.from(new Set([0, ...solvedBudgets])).sort(
      (a, b) => a - b
    )

    const proportions = budgetPoints.map(budget => {
      const solvedCount = pairs.filter(
        pair => pair.solved && pair.budget <= budget
      ).length
      return pairs.length ? solvedCount / pairs.length : 0
    })

    return {
      algorithm: optimizer,
      budget: budgetPoints,
      proportion: proportions
    }
  })
}

export const fetchBenchmarkData = async (
  path: string
): Promise<BenchmarkDataSchema> => {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`Failed to load benchmark data from ${path}`)
  }
  return (await response.json()) as BenchmarkDataSchema
}
