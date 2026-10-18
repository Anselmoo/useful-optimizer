/* Generated from benchmarks/models.py by benchmarks/export_schema.py; do not edit. */

export type SchemaVersion = 2;
export type Generated = string;
export type GitSha = string | null;
export type Python = string;
export type Numpy = string;
export type Scipy = string;
/**
 * Evaluations per dimension for every run
 */
export type BudgetPerDim = number;
/**
 * Target precisions f - f_opt, descending
 */
export type Targets = number[];
export type NRuns = number;
export type Dimensions = number[];
export type Name = string;
export type Lower = number;
export type Upper = number;
export type FOpt = number;
export type Functions = FunctionInfo[];
/**
 * Optimizer every other one is tested against
 */
export type Baseline = string;
export type EcdfPoints = number;
/**
 * Errors below this tie in tests and ranks
 */
export type PrecisionFloor = number;
export type TimeCapSeconds = number | null;
export type CellTimeSeconds = number | null;
/**
 * Statistical procedure, in words
 */
export type Test = string;
/**
 * Library optimizers not part of the suite
 */
export type Excluded = string[];
export type Name1 = string;
export type Category = string;
export type Baseline1 = boolean;
/**
 * Docs page path, if the optimizer has one
 */
export type Page = string | null;
export type Optimizers = OptimizerInfo[];
export type Function = string;
export type Dim = number;
export type File = string;
export type Optimizers1 = number;
export type Cells = CellRef[];
export type Function1 = string;
export type Dim1 = number;
export type Budget = number;
/**
 * Evaluation counts of the curve points
 */
export type Checkpoints = number[];
export type Targets1 = number[];
/**
 * Evaluations per dimension of the ECDF points
 */
export type EcdfBudgets = number[];
/**
 * Runs with a result
 */
export type Runs = number;
export type RunsPlanned = number;
export type Failed = number;
/**
 * Runs cut off by the wall-clock cap
 */
export type Truncated = number;
/**
 * Runs that used the whole evaluation budget
 */
export type Exhausted = number;
export type MaxIter = number;
/**
 * Final precision of every usable run
 */
export type Finals = number[];
export type Mean = number;
export type Std = number;
export type Median = number;
export type Q1 = number;
export type Q3 = number;
export type Min = number;
export type Max = number;
/**
 * Success rate per target
 */
export type Success = (number | null)[];
/**
 * COCO expected running time per target
 */
export type Ert = (number | null)[];
/**
 * COCO SP1 per target
 */
export type Sp1 = (number | null)[];
/**
 * Fraction of (run, target) pairs reached
 */
export type Ecdf = number[] | null;
export type Q11 = (number | null)[];
export type Median1 = (number | null)[];
export type Q31 = (number | null)[];
/**
 * Median normalised spread of recently evaluated points at the checkpoints
 */
export type Dispersion = (number | null)[] | null;
export type MedianRunSeconds = number;
export type MedianUsPerEvaluation = number;
/**
 * Raw one-sided p-value
 */
export type P = number;
/**
 * Holm-adjusted within the function and dimension
 */
export type PHolm = number;
export type U = number;
/**
 * Cliff's delta, +1 = every run beat every baseline run
 */
export type Delta = number;
export type SchemaVersion1 = 2;
export type Blocks = string[];
export type Name2 = string;
export type Category1 = string;
export type MeanRank = number;
export type Entries = RankEntry[];
export type Chi2 = number;
export type P1 = number;
export type Blocks1 = number;
export type Optimizers3 = number;
export type Function2 = string;
export type Seed = number;

/**
 * Published benchmark data, schema version 2. Generated from benchmarks/models.py by benchmarks/export_schema.py; do not edit.
 */
export interface BenchmarkBundle {
  index: Index;
  cell: Cell;
  ranks: Ranks;
  trajectory: Trajectory;
}
/**
 * Entry point of the benchmark data: ``index.json``.
 */
export interface Index {
  schema_version?: SchemaVersion;
  generated: Generated;
  git_sha: GitSha;
  versions: Versions;
  protocol: Protocol;
  optimizers: Optimizers;
  cells: Cells;
}
/**
 * Software versions the data was produced with.
 */
export interface Versions {
  python: Python;
  numpy: Numpy;
  scipy: Scipy;
}
/**
 * How the data was produced; everything a reader needs to interpret it.
 */
export interface Protocol {
  budget_per_dim: BudgetPerDim;
  targets: Targets;
  n_runs: NRuns;
  dimensions: Dimensions;
  functions: Functions;
  baseline: Baseline;
  ecdf_points: EcdfPoints;
  precision_floor: PrecisionFloor;
  time_cap_seconds: TimeCapSeconds;
  cell_time_seconds: CellTimeSeconds;
  test: Test;
  excluded: Excluded;
}
/**
 * A benchmark function with its search domain and known optimum.
 */
export interface FunctionInfo {
  name: Name;
  lower: Lower;
  upper: Upper;
  f_opt: FOpt;
}
/**
 * One benchmarked optimizer.
 */
export interface OptimizerInfo {
  name: Name1;
  category: Category;
  baseline?: Baseline1;
  page?: Page;
}
/**
 * Pointer to a cell file.
 */
export interface CellRef {
  function: Function;
  dim: Dim;
  file: File;
  optimizers: Optimizers1;
}
/**
 * One function and dimension: ``cells/<function>-<dim>.json``.
 */
export interface Cell {
  function: Function1;
  dim: Dim1;
  budget: Budget;
  checkpoints: Checkpoints;
  targets: Targets1;
  ecdf_budgets: EcdfBudgets;
  optimizers: Optimizers2;
}
export interface Optimizers2 {
  [k: string]: OptimizerCell;
}
/**
 * All metrics of one optimizer on one function and dimension.
 */
export interface OptimizerCell {
  runs: Runs;
  runs_planned: RunsPlanned;
  failed: Failed;
  truncated: Truncated;
  exhausted: Exhausted;
  max_iter: MaxIter;
  finals: Finals;
  stats: Stats | null;
  success: Success;
  ert: Ert;
  sp1: Sp1;
  ecdf: Ecdf;
  /**
   * Precision at the checkpoints
   */
  convergence: Curve | null;
  dispersion: Dispersion;
  time: TimeStats | null;
  /**
   * None for the baseline itself
   */
  test: BaselineTest | null;
}
/**
 * Distribution of the final precision over runs.
 */
export interface Stats {
  mean: Mean;
  std: Std;
  median: Median;
  q1: Q1;
  q3: Q3;
  min: Min;
  max: Max;
}
/**
 * Median and quartiles over runs at every checkpoint.
 */
export interface Curve {
  q1: Q11;
  median: Median1;
  q3: Q31;
}
/**
 * Wall-clock cost.
 */
export interface TimeStats {
  median_run_seconds: MedianRunSeconds;
  median_us_per_evaluation: MedianUsPerEvaluation;
}
/**
 * One-sided Mann-Whitney U test "better than the baseline" and its effect size.
 */
export interface BaselineTest {
  p: P;
  p_holm: PHolm;
  u: U;
  delta: Delta;
}
/**
 * Average ranks: ``ranks.json``.
 */
export interface Ranks {
  schema_version?: SchemaVersion1;
  blocks: Blocks;
  entries: Entries;
  friedman: Friedman | null;
}
/**
 * Rank of one optimizer.
 */
export interface RankEntry {
  name: Name2;
  category: Category1;
  mean_rank: MeanRank;
  ranks: Ranks1;
}
/**
 * Rank per block, key '<function>-<dim>'
 */
export interface Ranks1 {
  [k: string]: number;
}
/**
 * Friedman test over the blocks.
 */
export interface Friedman {
  chi2: Chi2;
  p: P1;
  blocks: Blocks1;
  optimizers: Optimizers3;
}
/**
 * Best-so-far path of the first run of every optimizer on a 2D function.
 */
export interface Trajectory {
  function: Function2;
  seed: Seed;
  optimizers: Optimizers4;
}
/**
 * Per optimizer rows [evaluations, x, y, f]
 */
export interface Optimizers4 {
  [k: string]: number[][];
}
