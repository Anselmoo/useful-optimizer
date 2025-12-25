# Benchmark History Tracking

This page documents the minimal history tracking and export format used by the benchmark suite.

## Overview

- The benchmark runner enables per-iteration **convergence history** when an optimizer is run with `track_history=True`.
- The convergence history is exported in the run record as the `convergence_history` key: a list of numeric values representing the best fitness at each recorded iteration (length expected to be `max_iter + 1`).
- For memory-sensitive runs, only `convergence_history` (best fitness per iteration) is recorded by default. Tracking full population snapshots is opt-in and memory intensive.

## Example: run a single benchmark with history

```bash
# Minimal single-run, prints result including 'convergence_history'
uv run python -c "from benchmarks.run_benchmark_suite import run_single_benchmark; print(run_single_benchmark('AntColony', 'shifted_ackley', -32.768, 32.768, dim=2, max_iter=10, seed=42))"
```

Expected output snippet:

```json
{
  "optimizer": "AntColony",
  "best_fitness": 0.0023412,
  "best_solution": [0.0012, -0.0023],
  "elapsed_time": 0.032,
  "convergence_history": [15.234, 12.156, 7.45, ..., 0.0023412],
  "status": "success"
}
```

## Notes

- `convergence_history` length should be `max_iter + 1` because the history records the initial state (iteration 0) and final state (iteration `max_iter`).
- For reproducibility, use the `seed` parameter when running benchmarks and include environment info (Python and NumPy versions) in the run metadata.

## Validation

- The Pydantic model `benchmarks.models.Run` includes `convergence_history: list[float] | None` for validation and schema generation.
- The benchmark integration test runs a small subset and asserts `convergence_history` is present and `len(convergence_history) == max_iter + 1`.
