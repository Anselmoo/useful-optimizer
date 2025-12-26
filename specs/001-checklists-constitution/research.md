# research.md — Phase 0 research notes

## Open questions & decisions to make

1. Compression format

   - Options: gzip (gz), zstd
   - Trade-offs: gz ubiquitous and simple; zstd gives better compression for large artifacts but GH Actions runner may need `zstd` installed.
   - Proposed: default gz, optional zstd if available (documented in quickstart).

2. Artifact schema fields

   - Required fields: `run_id`, `commit_sha`, `function`, `dim`, `seed`, `bounds`, `best_solution`, `best_fitness`, `history` (best_fitness per iteration), `runtime_seconds`, `max_memory_bytes`.
   - Validation: pydantic model + JSON Schema file at `schemas/benchmark-data-schema.json`.

3. Quick test parametrization

   - Quick tests: max_iter 10–100 (depending on algorithm), population_size small (e.g., min(10, 10\*dim)), single seed.
   - Target: complete in <1s on GH runner.

4. CI budget and scheduling

   - Quick PR checks: run on each PR using `benchmark_quick` marker.
   - Nightly full runs: scheduled nightly or weekly based on cost; upload artifacts and notify maintainers.

5. Doc generation flow

   - A job consumes artifacts and produces `docs/benchmarks/generated/<run-id>.md` with summary tables and small plots (optionally produced with ECharts or Matplotlib).

6. Rollback and gating
   - Docs generation is gated behind a flag until we have consistent nightly runs.
   - Merge strategy: single PR; if any failing nightly jobs, revert the merge or push a fix PR.

## Next research tasks (actionable)

- Implement a small prototype runner for `sphere` with gz output and a minimal schema; validate quick test runs on GH locally.
- Benchmark memory usage on a small run and record memory consumption to set memory goal thresholds.
- Prototype docs generation script that consumes a sample artifact and renders a small markdown snippet.
