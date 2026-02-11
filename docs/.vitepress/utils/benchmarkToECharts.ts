/**
 * Utility functions for transforming benchmark data to ECharts options
 */

import type { EChartsOption } from 'echarts'
import type {
  Run,
  Benchmarks,
  ConvergenceData,
  ECDFData,
  ViolinData
} from './types'

/**
 * Transform benchmark runs to convergence chart data
 * 
 * @param benchmarks - Benchmark results for a specific optimizer/function/dimension
 * @param algorithmName - Name of the algorithm for display
 * @returns Convergence data formatted for ConvergenceChart component
 */
export function toConvergenceData(
  benchmarks: Benchmarks,
  algorithmName: string
): ConvergenceData {
  const runs = benchmarks.runs.filter(run => run.history?.best_fitness)
  
  if (runs.length === 0) {
    return {
      algorithm: algorithmName,
      iterations: [],
      mean: [],
      std: []
    }
  }

  // Get the convergence history from all runs
  const allHistories = runs.map(run => run.history!.best_fitness!)
  const maxLength = Math.max(...allHistories.map(h => h.length))
  
  // Calculate mean and std at each iteration
  const iterations: number[] = []
  const mean: number[] = []
  const std: number[] = []
  
  for (let i = 0; i < maxLength; i++) {
    const valuesAtIteration = allHistories
      .filter(h => i < h.length)
      .map(h => h[i])
    
    if (valuesAtIteration.length > 0) {
      iterations.push(i)
      
      // Calculate mean
      const meanVal = valuesAtIteration.reduce((a, b) => a + b, 0) / valuesAtIteration.length
      mean.push(meanVal)
      
      // Calculate standard deviation
      if (valuesAtIteration.length > 1) {
        const variance = valuesAtIteration
          .map(v => Math.pow(v - meanVal, 2))
          .reduce((a, b) => a + b, 0) / valuesAtIteration.length
        std.push(Math.sqrt(variance))
      } else {
        std.push(0)
      }
    }
  }
  
  return {
    algorithm: algorithmName,
    iterations,
    mean,
    std
  }
}

/**
 * Transform multiple benchmark results to convergence chart data
 * 
 * @param benchmarksMap - Map of optimizer name to benchmark results
 * @returns Array of convergence data for multiple algorithms
 */
export function toMultiConvergenceData(
  benchmarksMap: Record<string, Benchmarks>
): ConvergenceData[] {
  return Object.entries(benchmarksMap).map(([name, benchmarks]) =>
    toConvergenceData(benchmarks, name)
  )
}

/**
 * Calculate ECDF (Empirical Cumulative Distribution Function) from benchmark runs
 * 
 * @param benchmarks - Benchmark results
 * @param algorithmName - Name of the algorithm
 * @param targetPrecisions - Target fitness thresholds to evaluate (e.g., [1e-1, 1e-3, 1e-5])
 * @param dimension - Problem dimension for budget calculation
 * @returns ECDF data formatted for ECDFChart component
 */
export function toECDFData(
  benchmarks: Benchmarks,
  algorithmName: string,
  targetPrecisions: number[] = [1e-1, 1e-3, 1e-5, 1e-7],
  dimension: number = 2
): ECDFData {
  const runs = benchmarks.runs.filter(run => run.history?.best_fitness)
  
  if (runs.length === 0) {
    return {
      algorithm: algorithmName,
      budget: [],
      proportion: []
    }
  }

  // Calculate budget points (function evaluations / dimension)
  const maxEvals = Math.max(...runs.map(r => r.n_evaluations))
  const budgetPoints = Array.from({ length: 20 }, (_, i) => 
    Math.pow(10, Math.log10(10) + (Math.log10(maxEvals / dimension) - Math.log10(10)) * i / 19)
  )
  
  const budget: number[] = []
  const proportion: number[] = []
  
  // For each budget point, calculate proportion of (function, target) pairs solved
  for (const budgetVal of budgetPoints) {
    const absoluteBudget = budgetVal * dimension
    
    // Count how many runs achieved each target precision within this budget
    let totalTargetsReached = 0
    const totalTargets = targetPrecisions.length
    
    for (const target of targetPrecisions) {
      const runsReachingTarget = runs.filter(run => {
        const history = run.history!.best_fitness!
        // Find if any point in history reached the target within budget
        const evaluationsPerIteration = run.n_evaluations / history.length
        for (let i = 0; i < history.length; i++) {
          const evals = (i + 1) * evaluationsPerIteration
          if (evals <= absoluteBudget && history[i] <= target) {
            return true
          }
        }
        return false
      })
      
      if (runsReachingTarget.length > 0) {
        totalTargetsReached++
      }
    }
    
    budget.push(budgetVal)
    proportion.push(totalTargetsReached / totalTargets)
  }
  
  return {
    algorithm: algorithmName,
    budget,
    proportion
  }
}

/**
 * Transform multiple benchmark results to ECDF data
 * 
 * @param benchmarksMap - Map of optimizer name to benchmark results
 * @param targetPrecisions - Target fitness thresholds
 * @param dimension - Problem dimension
 * @returns Array of ECDF data for multiple algorithms
 */
export function toMultiECDFData(
  benchmarksMap: Record<string, Benchmarks>,
  targetPrecisions: number[] = [1e-1, 1e-3, 1e-5, 1e-7],
  dimension: number = 2
): ECDFData[] {
  return Object.entries(benchmarksMap).map(([name, benchmarks]) =>
    toECDFData(benchmarks, name, targetPrecisions, dimension)
  )
}

/**
 * Transform benchmark runs to violin plot data
 * 
 * @param benchmarks - Benchmark results
 * @param algorithmName - Name of the algorithm
 * @returns Violin plot data formatted for ViolinPlot component
 */
export function toViolinData(
  benchmarks: Benchmarks,
  algorithmName: string
): ViolinData {
  return {
    algorithm: algorithmName,
    values: benchmarks.runs.map(run => run.best_fitness)
  }
}

/**
 * Transform multiple benchmark results to violin plot data
 * 
 * @param benchmarksMap - Map of optimizer name to benchmark results
 * @returns Array of violin plot data for multiple algorithms
 */
export function toMultiViolinData(
  benchmarksMap: Record<string, Benchmarks>
): ViolinData[] {
  return Object.entries(benchmarksMap).map(([name, benchmarks]) =>
    toViolinData(benchmarks, name)
  )
}

/**
 * Helper function to calculate statistical measures
 */
export function calculateStats(values: number[]) {
  if (values.length === 0) {
    return {
      mean: 0,
      std: 0,
      min: 0,
      max: 0,
      median: 0,
      q1: 0,
      q3: 0
    }
  }

  const sorted = [...values].sort((a, b) => a - b)
  const n = sorted.length
  
  const mean = values.reduce((a, b) => a + b, 0) / n
  const variance = values.map(v => Math.pow(v - mean, 2)).reduce((a, b) => a + b, 0) / n
  const std = Math.sqrt(variance)
  
  const median = sorted[Math.floor(n * 0.5)]
  const q1 = sorted[Math.floor(n * 0.25)]
  const q3 = sorted[Math.floor(n * 0.75)]
  const min = sorted[0]
  const max = sorted[n - 1]
  
  return { mean, std, min, max, median, q1, q3 }
}
