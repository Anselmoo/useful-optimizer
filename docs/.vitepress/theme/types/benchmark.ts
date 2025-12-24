// Auto-generated from benchmark-data-schema.json - DO NOT EDIT

/**
 * IOHprofiler-compatible benchmark results for optimizer evaluation
 */
export interface BenchmarkDataSchema {
  metadata: {
    max_iterations: number;
    n_runs: number;
    dimensions: number[];
    timestamp: string;
    python_version?: string;
    numpy_version?: string;
    [k: string]: unknown;
  };
  benchmarks: {
    /**
     * Function name (e.g., 'shifted_ackley') → dimension → optimizer → results
     */
    [k: string]: {
      /**
       * Dimension (e.g., '2', '10', '30')
       */
      [k: string]: {
        /**
         * Optimizer name
         */
        [k: string]: {
          runs: {
            best_fitness: number;
            best_solution: number[];
            n_evaluations: number;
            history?: {
              best_fitness?: number[];
              mean_fitness?: number[];
              [k: string]: unknown;
            };
            [k: string]: unknown;
          }[];
          statistics: {
            mean_fitness: number;
            std_fitness: number;
            min_fitness: number;
            max_fitness: number;
            median_fitness: number;
            q1_fitness?: number;
            q3_fitness?: number;
            [k: string]: unknown;
          };
          success_rate: number;
          [k: string]: unknown;
        };
      };
    };
  };
  [k: string]: unknown;
}
