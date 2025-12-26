# Discovery Summary — Replace trivial doctests with spec-driven benchmarks

Date: 2025-12-26
Repo: useful-optimizer

Summary:
- Total doctest-like snippets found (quick scan): ~200 matches across `opt/` and docs.
- Common patterns: trivial shape checks (e.g., `len(solution) == 10`), simple `isinstance` checks, and a small number of examples already using `optimizer.benchmark(store=True)` in templates and specs.

Top candidate files (see `files.json`):
- `opt/constrained/penalty_method.py` (several trivial checks)
- `opt/swarm_intelligence/manta_ray.py`
- `opt/gradient_based/adadelta.py`
- `opt/metaheuristic/variable_depth_search.py`
- `opt/benchmark/functions.py` (doctest examples for benchmark functions)
- `opt/abstract/multi_objective.py` (multi-objective examples)
- `specs/002-replace-doctests-with-spec-driven-benchmarks/spec.md` (already contains the in-spec minimal example)

Immediate recommendations:
1. Replace trivial shape/assert doctests with a mini-benchmark snippet using `optimizer.benchmark(store=True)` and a JSON schema validation step.
2. Implement `export_benchmark_json` and `validate_benchmark_json` helpers early so replacements can rely on stable export/validation behavior.
3. Add `benchmark_quick` tests parametrized per optimizer to run as PR checks.

Next steps (Discovery outputs):
- `discovery/files.json` (indexed candidate files and samples)
- Proceed to Gap & Dependency Analysis
