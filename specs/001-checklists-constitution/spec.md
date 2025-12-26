# Feature Spec: Replace doctests with spec-driven benchmark tests

## Overview

Replace selected inline doctests in optimizer modules with spec-driven benchmark tests that generate reproducible benchmark artifacts used to produce documentation examples and to power CI validation.

## Motivation

- Doctests are brittle and may not reflect real benchmark outputs.
- Generating real benchmark artifacts provides richer examples for docs and enables regression detection.

## User Stories

- Given a developer editing an optimizer, when they run the quick benchmark, then they get a small validated artifact and a pass/fail outcome in <1s.
- Given the maintainer, when a PR is merged, then nightly scheduled full-run jobs produce full BBOB/COCO artifacts and upload them for long-term analysis.

## Acceptance Criteria

- Quick benchmark tests exist under `tests/benchmark_quick` and run <1s on GH runners.
- Every replaced doctest is covered by a spec file and a quick test.
- Artifacts validate against `schemas/benchmark-data-schema.json`.
- Docs pages are generated from artifacts and included as draft PR content in the main PR.

## How to run locally

- Quick: `uv run pytest -m benchmark_quick -q`
- Full: `uv run python -m opt.benchmark.runner --function sphere --dim 10 --seed 0 --out=benchmarks/results/myrun` (large runs may be skipped locally)

## Edge cases

- Non-deterministic functions must be stabilized via seed and documented.
- Very long-running functions should be excluded from quick tests and mocked or run in nightly only.

## Notes

- Start migration with `MantaRayForagingOptimization` as canonical example.
