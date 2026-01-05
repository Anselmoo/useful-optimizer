/**
 * TypeScript type definitions for benchmark data structures
 * Matches the Pydantic models from benchmarks/models.py
 */

/**
 * Metadata for benchmark execution
 */
export interface BenchmarkMetadata {
  max_iterations: number
  n_runs: number
  dimensions: number[]
  timestamp: string
  python_version?: string
  numpy_version?: string
}

/**
 * History tracking data for optimization runs
 */
export interface History {
  best_fitness?: number[]
  mean_fitness?: number[]
}

/**
 * Individual optimization run results
 */
export interface Run {
  best_fitness: number
  best_solution: number[]
  n_evaluations: number
  history?: History
}

/**
 * Statistical summary of benchmark results
 */
export interface Statistics {
  mean_fitness: number
  std_fitness: number
  min_fitness: number
  max_fitness: number
  median_fitness: number
  q1_fitness?: number
  q3_fitness?: number
}

/**
 * Benchmark results for a specific optimizer/function/dimension combination
 */
export interface Benchmarks {
  runs: Run[]
  statistics: Statistics
  success_rate: number
}

/**
 * Complete benchmark data schema
 * Structure: benchmarks[function][dimension][optimizer]
 */
export interface BenchmarkDataSchema {
  metadata: BenchmarkMetadata
  benchmarks: {
    [functionName: string]: {
      [dimension: string]: {
        [optimizerName: string]: Benchmarks
      }
    }
  }
}

/**
 * Convergence data for ECharts
 */
export interface ConvergenceData {
  algorithm: string
  iterations: number[]
  mean: number[]
  std?: number[]
  min?: number[]
  max?: number[]
}

/**
 * ECDF data for ECharts
 */
export interface ECDFData {
  algorithm: string
  budget: number[]      // Function evaluations / dimension
  proportion: number[]  // Proportion of (function, target) pairs solved
}

/**
 * Violin plot data for ECharts
 */
export interface ViolinData {
  algorithm: string
  values: number[]  // Fitness values from multiple runs
}
