# Architecture — Benchmarks & Mini-Benchmarks

This document describes the high-level architecture for implementing spec-driven benchmark mini-examples, quick PR checks, and nightly full-run artifacts for the useful-optimizer project.

## Components

- AbstractOptimizer/AbstractMultiObjectiveOptimizer
  - Add `benchmark(store=True, out_path=None, **kwargs)` method that runs a short internal benchmark and returns `{'path': ..., 'metadata': {...}}`.
- Export Helper
  - `opt/benchmark/utils.py` provides `export_benchmark_json(data, path, schema)` and `validate_benchmark_json(path, schema)`.
- CI Quick Job
  - Runs `benchmark_quick` tests on PRs and validates produced JSON artifacts.
- CI Nightly Job
  - Runs `benchmark_full` tests, compresses artifacts, runs schema validation, and archives artifacts to project artifact store.
- Pre-commit Hooks
  - Validate changed `.json` artifacts (via `validate-benchmark-json`) and detect forbidden patterns (e.g., `mcp_` in `opt/` files)
- Serena Memory
  - All MCP calls used in the plan and execution (discovery, gap analysis, plan validation) are recorded with `mcp_serena_write_memory` for auditability.

## Data Flow (summary)

Developer -> GitHub PR -> CI Quick job runs `benchmark_quick` tests -> `AbstractOptimizer.benchmark()` or test harness runs short benchmark -> `export_benchmark_json` writes artifact to `benchmarks/output/` -> `validate_benchmark_json` validates schema -> success/failure reported.

Nightly: Scheduler -> CI Nightly -> run `benchmark_full` tests -> compress artifacts -> `validate_benchmark_json` -> upload to artifact store and signal retention/archival pipeline.

## Diagrams

See `architecture/diagrams.mmd` for Mermaid sources and `architecture/` for generated images.

## Acceptance Criteria (architecture-level)

- All components have clear responsibilities and authored interfaces.
- `benchmark()` and `export_benchmark_json()` have well-documented signatures and docstrings.
- CI jobs and pre-commit hooks are defined with exact commands and pass on a test branch.
- At least one Mermaid architecture diagram and one CI choreography diagram exist in `architecture/`.
