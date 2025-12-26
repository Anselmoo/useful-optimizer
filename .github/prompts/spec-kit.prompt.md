## Essential commands for the Spec-Driven Development workflow:

- /speckit.constitution Create or update project governing principles and development guidelines

  Answer: Use the Spec Kit constitution template (fetch via Context7) and produce a short, actionable _Constitution_ document (saved at `.github/CONSTITUTION.md`) that explicitly enforces the following rules and processes. The constitution must be generated via a prompt to `mcp_context7_get-library-docs` (library `/github/spec-kit`) and validated with `mcp_ai-agent-guid_guidelines-validator`.

  Required constitution sections (these must appear verbatim in the output):

  - **Principles**: Test-First (TDD), Spec-Driven Development, Reproducible Benchmarks (seeded, archived), Minimal Doctests (replace with mini-benchmarks), Memory-Efficient History Tracking (track_history opt-in), and CI-enforced spec checks.
  - **Enforcement**: All PRs modifying optimizers must include a spec update, at least one `benchmark_quick` test, and a passing `validate-benchmark-json` pre-commit check or CI job.
  - **MCP Usage**: All spec, plan, and implementation steps must call (and log) these MCPs: `mcp_context7_get-library-docs`, `mcp_ai-agent-guid_prompt-chaining-builder`, `mcp_ai-agent-guid_guidelines-validator`, `mcp_serena_*` (activate/find/insert/replace where needed), and GitHub MCPs used to create branches and PRs.
  - **Artifacts & Versioning**: Benchmark artifacts must be stored as `benchmarks/output/<algorithm>-<timestamp>-s<seed>.json` and archived by nightly CI. Schema version must be recorded in the JSON metadata (`schema_version`).
  - **Review Criteria**: PRs must include a short gap-analysis (generated via `mcp_ai-agent-guid_gap-frameworks-analyzers`) and a checklist referencing `.github/checklists/benchmarks-spec.md`.

  Example generator prompt (use verbatim):

  "Using the Spec Kit constitution template (library `/github/spec-kit`), draft a short Constitution file for the 'useful-optimizer' project: include Principles, Enforcement rules, MCP usage requirements, Artifact naming/versioning rules, and Review Criteria (3-5 bullets each). Validate the text using `mcp_ai-agent-guid_guidelines-validator` to ensure it follows repository doc standards and output a markdown file content ready to save at `.github/CONSTITUTION.md`."

  Expected output: a markdown document `.github/CONSTITUTION.md` (3-8 short sections) that will be checked into the repo. Include example lines that reviewers can grep for, e.g. `Reproducible Benchmarks: seeds 0-14 required for full runs`.

  Local verification steps:

  1. Run `mcp_context7_resolve-library-id` to locate `/github/spec-kit` and then `mcp_context7_get-library-docs` to fetch the template.
  2. Call `mcp_ai-agent-guid_guidelines-validator` on the generated markdown to get a validation score and suggestions.
  3. Save to `.github/CONSTITUTION.md` and add a small test that verifies the file contains the 'Principles' heading.

  Required MCPs for this step: `mcp_context7_resolve-library-id`, `mcp_context7_get-library-docs`, `mcp_ai-agent-guid_guidelines-validator`, `mcp_serena_write_memory` (optionally to store the generated constitution for traceability).

- /speckit.specify Define what you want to build (requirements and user stories)

  Answer: Use the Spec Kit `spec-template` (resolve `/github/spec-kit`) to write precise feature specs that _replace trivial doctests with reproducible, machine-parseable benchmark examples_. The feature spec MUST include the following concrete items (use the examples below as templates):

  - **Acceptance criteria (must be verifiable by CI):**

    - Quick sanity benchmark (`benchmark_quick`) that runs < 1s on CI (small `dim`, explicit `early_stop`/`stop_threshold` usage, and a conservative `max_iter` such as `<= 500`) and _writes_ an artifact in JSON that conforms to `docs/schemas/benchmark-data-schema.json`. Benchmarks MUST prefer early stopping over relying on very large `max_iter` values; **do not use** `max_iter=10000` as a default in benchmark comparisons without a documented justification.
    - Deterministic runs with `seed` documented and checked (e.g., `metadata.seed == 42`).
    - Full nightly benchmark (`benchmark_full`) that reproduces longer runs and archives artifacts; full runs MUST also include early-stop support so long-running jobs terminate once the solver converges to the threshold.
    - Tests assert that the generated JSON validates against the project's schema (no manual inspection).

  - **Implementation targets:**

    - Update `AbstractOptimizer` and `AbstractMultiObjectiveOptimizer` to expose a `benchmark(..., store=True, out_path: str|None)` method that runs an internal short benchmark, saves artifacts and returns a dict with `path` and `metadata`.
    - Add `export_benchmark_json(path, schema='docs/schemas/benchmark-data-schema.json')` helper for explicit export and validation.
    - Add an `early_stop` (or `stop_threshold`) parameter to `AbstractOptimizer`/`AbstractMultiObjectiveOptimizer` and ensure all optimizer implementations support early stopping based on improvement thresholds or plateau detection. Benchmark metadata MUST include `iterations` (int), `stopped_early` (bool), and `stopping_reason` (str).
    - Audit the codebase for usages of very large `max_iter` (e.g., `max_iter=10000`) and replace with conservative defaults or add documented justification where long runs are required for research; add pre-commit checks that flag unexplained `max_iter>=5000` literals in tests or docs.
    - Replace trivial doctest examples in docstrings with _mini benchmark examples_ demonstrating `optimizer.benchmark(store=True)` and a small JSON validation check.

  - **Example prompt (use verbatim for spec generation):**

    "Write a Feature Specification to replace trivial doctest examples with spec-driven benchmark tests. The spec must provide GIVEN/WHEN/THEN scenarios, exact CLI/test commands to run locally (e.g., `uv run pytest -q -k benchmark_quick`), exact JSON schema to validate (`docs/schemas/benchmark-data-schema.json`), and sample benchmark example code. Include edge cases (missing fields, schema mismatch) and 'How to validate locally' steps. Use `mcp_context7_get-library-docs` and `mcp_ai-agent-guid_prompt-chaining-builder` to produce the spec draft."

  - **Minimal in-spec example to include (sanity test):**

    ```py
    >>> from opt.benchmark.functions import sphere
    >>> opt = SimulatedAnnealing(func=sphere, lower_bound=-5, upper_bound=5, dim=2, max_iter=100, early_stop=1e-6, seed=42)
    >>> result = opt.benchmark(store=True)
    >>> import jsonschema, json
    >>> with open(result['path']) as f: data = json.load(f)
    >>> jsonschema.validate(instance=data, schema_path='docs/schemas/benchmark-data-schema.json')  # should not raise
    >>> assert data['metadata']['iterations'] <= 100
    >>> assert isinstance(data['metadata'].get('stopped_early'), bool)
    True
    ```

  **Tools & MCPs (required when generating the spec):**

  - `mcp_context7_get-library-docs` (fetch Spec Kit and COCO docs)
  - `mcp_ai-agent-guid_prompt-chaining-builder` (generate stepwise specs)
  - `mcp_ai-agent-guid_documentation-generator-prompt-builder` (doc examples)
  - `mcp_ai-agent-guid_guidelines-validator` (validate docstring & schema rules)

  Use these tools to produce a spec that is _self-contained_, machine-checkable, and lists the exact code/CI changes required (files, tests, pre-commit hook, new scripts).

  Tools: `mcp_context7_get-library-docs`, `mcp_ai-agent-guid_prompt-chaining-builder` to generate spec drafts.

- /speckit.plan Create technical implementation plans with your chosen tech stack

  Answer: Use `mcp_ai-agent-guid_architecture-design-prompt-builder` as the _orchestrator_ in a multi-step, MCP-rich planning workflow (Discovery → Architecture → Design → Implementation Plan → Validation & Rollout). For this complex feature, **do not** treat the architecture builder as a single one-shot call; instead perform a sequence of targeted MCP calls (discovery, gap analysis, security review, dependency audit, sprint scheduling, and validator loops). The plan must produce machine-readable artifacts and an audit trail of all MCP calls.

  Required multi-step workflow (detailed):

  1. Discovery & Context Capture (MCPs: `mcp_context7_get-library-docs`, `mcp_serena_find_file`, `mcp_serena_search_for_pattern`, `vscode-websearchforcopilot_webSearch`, `mcp_ai-agent-guid_code-analysis-prompt-builder`)

     - Goals: collect existing specs, docstrings, doctests, benchmark artifacts and CI configurations. Scan the repository for long `max_iter` usages (e.g., `max_iter=10000`) and for optimizers lacking early-stop/stop-threshold support; produce `discovery.md` summarizing current baseline and a file index of doctests and candidates for early-stop improvements.
     - Deliverables: `discovery.md`, `discovery/files.json` (paths and line ranges), and a short list of candidate doctests and optimizer implementations needing early-stop support.
     - MCP logging: record each call with `mcp_serena_write_memory` including parameters and a 1-sentence result summary.

  2. Gap & Dependency Analysis (MCPs: `mcp_ai-agent-guid_gap-frameworks-analyzers`, `mcp_ai-agent-guid_dependency-auditor`, `mcp_ai-agent-guid_iterative-coverage-enhancer`)

     - Goals: map missing tests, schema validation gaps, and risky dependencies (non-deterministic RNGs or use of global state). Provide prioritized gaps with impact estimates.
     - Deliverables: `.github/analysis/benchmarks-gap-report.md` (JSON + markdown summary) and `dependencies-report.md` listing packages and risky patterns.
     - Acceptance: gap report contains categories and at least top-5 actionable tasks.

  3. Security & Hardening Review (MCPs: `mcp_ai-agent-guid_security-hardening-prompt-builder`, `mcp_ai-agent-guid_guidelines-validator`)

     - Goals: identify data leakage risks, ensure artifacts do not contain secrets, and define pre-commit checks preventing sensitive information in artifacts. Validate documentation quality and compliance with project guidelines.
     - Deliverables: `security-review.md` and updated pre-commit checks list.

  4. High-Level Architecture & Design (MCPs: `mcp_ai-agent-guid_architecture-design-prompt-builder`, `mcp_ai-agent-guid_mermaid-diagram-generator`)

     - Goals: produce a design doc describing modules, public method contracts (`benchmark()` signature), artifact formats and storage paths, CI choreography, and failure modes.
     - Deliverables: `architecture.md` (includes diagrams in Mermaid), `architecture/diagrams.mmd` (mermaid source), and a decision log with trade-offs considered.
     - Acceptance: design includes explicit acceptance criteria and a schema for plan YAML frontmatter.

  5. Implementation Planning & Task Generation (MCPs: `mcp_ai-agent-guid_prompt-chaining-builder`, `mcp_ai-agent-guid_sprint-timeline-calculator`, `mcp_ai-agent-guid_prompt-flow-builder`)

     - Goals: turn architecture into `plan.md` and `tasks.md` with TIDs, dependencies, owners, and acceptance commands. Produce a `tasks.json` for automation.
     - Deliverables: `plan.md` (phase-level), `tasks.md` (task-level with TIDs), `tasks.json` (machine-readable), sprint schedule (per `mcp_ai-agent-guid_sprint-timeline-calculator`).
     - Acceptance: all tasks have owners, estimates, and acceptance commands; `tasks.json` is JSON-Schema valid.

  6. Validation & Policy Gates (MCPs: `mcp_ai-agent-guid_guidelines-validator`, `mcp_ai-agent-guid_iterative-coverage-enhancer`)

     - Goals: validate plan wording, check coverage, and ensure gates are clear. Iterate on plan until validator score ≥ 75/100 or else document rationale for accepting lower thresholds.
     - Deliverables: `plan-validation.md` with validator scores and iterations logged.

  7. Rollout & Runbook (MCPs: `mcp_ai-agent-guid_gap-frameworks-analyzers`, `mcp_ai-agent-guid_prompt-chaining-builder`)
     - Goals: prepare the 'one-big-PR' workflow instructions, smoke test commands, rollback commands, and monitoring checks for nightly jobs. Create a short runbook for emergency reversion and for re-running last-known-good artifacts.
     - Deliverables: `runbook.md`, `rollback-commands.md`, and `ci/archiving.md` reference.

  Cross-cutting requirements for all steps:

  - Every MCP call MUST be recorded in Serena memory (`mcp_serena_write_memory`) with parameters and a one-line summary.
  - Use `mcp_ai-agent-guid_mermaid-diagram-generator` to produce at least one diagram for architecture and one for CI choreography; include both mermaid source and a PNG/SVG artifact in `architecture/`.
  - Include a risk register in `plan.md` with mitigation tasks and probability×impact scoring.
  - Produce YAML frontmatter in all `plan.md` outputs containing `phase`, `estimate_points`, `owner`, `acceptance_tests` and `mcp_calls` array (with call ids and short result text) so downstream tasks and automation can parse the plan.

  Default success criteria for planning phase:

  - `plan.md` exists, `tasks.md` contains at least T001–T010 with owners and estimates, gap report has no high-impact unresolved items, and `mcp_ai-agent-guid_guidelines-validator` score ≥ 75/100 (or documented exception).

  Recommended MCP call pattern and sample calls (examples to include in plan text):

  - Discovery: `mcp_context7_get-library-docs({context7CompatibleLibraryID: '/github/spec-kit', mode: 'code', topic: 'spec-template'})`
  - Gap analysis: `mcp_ai-agent-guid_gap-frameworks-analyzers({currentState: 'repo snapshot', desiredState: 'spec-driven benchmarks', frameworks: ['testing','performance','process']})`
  - Performance & code analysis: `mcp_ai-agent-guid_code-analysis-prompt-builder({analysisType: 'performance', content: 'scan for missing early-stop and inefficient loop bounds like max_iter=10000'})`
  - Web research for early-stop best practices: `vscode-websearchforcopilot_webSearch({query: 'early stopping thresholds optimizer max_iter best practices'})`
  - Dependency audit: `mcp_ai-agent-guid_dependency-auditor({dependencyContent: 'contents of pyproject.toml', checkVulnerabilities: true})`
  - Architecture sketch: `mcp_ai-agent-guid_architecture-design-prompt-builder({systemRequirements: 'benchmark runner, export helper, CI jobs, performance gates', scale: 'small', technologyStack: 'python', includeMermaid: true})`
  - Plan validation loop: `mcp_ai-agent-guid_guidelines-validator({practiceDescription: 'Candidate plan content', category: 'workflow'})`

  Output: a machine-parseable `plan.md` (with YAML frontmatter), `architecture.md` (with diagrams), `discovery.md`, `benchmarks-gap-report.md`, `tasks.md`, `tasks.json`, `runbook.md`, and a `plan-validation.md` file logging validator scores and MCP calls. Ensure every file lists the MCP calls used and stores the same short call summary in Serena memory for auditability.

- /speckit.tasks Generate actionable task lists for implementation

  Answer: Use `mcp_ai-agent-guid_prompt-chaining-builder` to expand the plan into prioritized, dependency-ordered tasks and produce a `tasks.md` with the following rules and features:

  - Task identifier convention: `T###` (zero-padded sequential numbers). Every task must include: `[ ] [T###] [P?] [Story?] Short description — Owner: @username — Est: X pts — Files: path/to/file` where `[P?]` denotes parallelizable and `[Story?]` maps to a user story label like `[US1]`.
  - Task fields and acceptance: each task must include exact acceptance criteria (commands or tests to run), a list of dependent TIDs (if any), estimated story points, and the expected artifact(s) produced.
  - Task categories (must produce separate sections in `tasks.md`): Setup, Foundational, User Stories (by priority), CI & Pre-commit, Docs, Polish & Release, Rollback/Runbook tasks.
  - Sample prioritized task list (example):

    - [ ] [T001] [ ] [Setup] Create `opt/benchmark/utils.py` and implement `export_benchmark_json` — Owner: @you — Est: 2 pts — Files: `opt/benchmark/utils.py` — Acceptance: `python -c "from opt.benchmark.utils import export_benchmark_json"` and unit tests in `tests/test_benchmark_export.py` pass.
    - [ ] [T002] [ ] [Foundational] Add `benchmark()` to `opt/abstract_optimizer.py` (and mirror in multi-objective) — Owner: @you — Est: 4 pts — Files: `opt/abstract_optimizer.py`, `opt/multi_objective/abstract_multi_objective.py` — Acceptance: `uv run pytest -q tests/test_benchmark_export.py -q`.
    - [ ] [T002.1] [ ] [Foundational] Add early stopping support: add `early_stop`/`stop_threshold` parameter to `AbstractOptimizer` and implement in concrete optimizers; ensure benchmark metadata includes `iterations`, `stopped_early`, and `stopping_reason` — Owner: @you — Est: 4 pts — Files: `opt/abstract/*`, `opt/**` — Acceptance: `uv run pytest -q -k early_stop` (new tests pass and `metadata['stopped_early']` is present).
    - [ ] [T002.2] [ ] [Foundational] Audit codebase for `max_iter=10000` occurrences and replace with conservative defaults or add documented justification; add pre-commit check to flag unexplained `max_iter>=5000` literals — Owner: @you — Est: 2 pts — Files: repo-wide — Acceptance: grep for `max_iter=10000` returns 0 results or has PR-level justification comments.
    - [ ] [T003] [P] [US1] Implement `scripts/replace_trivial_doctests.py` with `--dry-run` and `--apply` flags — Owner: @you — Est: 6 pts — Acceptance: dry-run outputs a report; unit test `tests/test_doctest_replacements.py` passes.
    - [ ] [T004] [ ] [CI] Add `.github/workflows/benchmarks-quick.yml` that runs `uv run pytest -q -k benchmark_quick` on PRs — Owner: @you — Est: 2 pts — Acceptance: CI job appears and passes in a test branch.
    - [ ] [T005] [ ] [CI] Add nightly `.github/workflows/benchmarks-full.yml` to run `@pytest.mark.benchmark_full` and upload artifacts to `benchmarks/output/` — Owner: @you — Est: 3 pts — Acceptance: artifacts are uploaded and validated against schema.
    - [ ] [T006] [ ] [Docs] Add docs and example snippets in `docs/` and update docstrings to include the in-spec minimal example — Owner: @docs — Est: 2 pts — Acceptance: `grep -R "benchmark(store=True)" docs/` finds examples; docs build passes.
    - [ ] [T007] [ ] [Pre-commit] Add `validate-benchmark-json` hook and `scripts/validate_benchmark_json.py` — Owner: @you — Est: 1 pt — Acceptance: pre-commit runs locally and fails on invalid JSON.

  - Prioritization & dependencies: mark tasks with `[P]` when they can be executed in parallel. The `mcp_ai-agent-guid_sprint-timeline-calculator` can be used to turn the ordered tasks into a sprint schedule for the single big PR (respecting dependencies and capacity constraints).
  - Automation & traceability requirements: every task that runs an MCP or modifies specs must include the MCP call in its description and call `mcp_serena_write_memory` to record the action (e.g., `mcp: mcp_ai-agent-guid_gap-frameworks-analyzers params: {...}`).
  - Output format: produce `tasks.md` following Spec Kit task-template format (checkboxes, TIDs, story labels). Export a small JSON version `tasks.json` with fields for programmatic consumption by automation scripts.

  Use these MCPs while generating tasks and log calls for auditability:

  - `mcp_ai-agent-guid_prompt-chaining-builder` (primary)
  - `mcp_ai-agent-guid_sprint-timeline-calculator` (optional for schedule generation)
  - `mcp_ai-agent-guid_gap-frameworks-analyzers` (to list actionable fixes for coverage gaps)
  - `mcp_serena_write_memory` (record generated tasks and MCP call logs)

  Local verification: ensure `tasks.md` is created in the feature folder, run a small script to verify each TID has an owner and an acceptance command, and that tasks.json is well-formed.

- /speckit.implement Execute all tasks to build the feature according to the plan

  Answer: Use the prompt chain from `mcp_ai-agent-guid_prompt-chaining-builder` to implement: create branch, write specs, add `benchmark_quick` tests (with JSON export), add `@pytest.mark.benchmark_full` tests (artifact archival), validate history tracking (PR #119), add CI workflows, and prepare a single comprehensive PR. Local dev steps: `uv sync`, run quick tests, lint/format, run full integration locally if needed, then push branch and open PR (one big update). Use `mcp_context7_resolve-library-id` and `mcp_context7_get-library-docs` to reference external spec templates, and the following Serena/GitHub MCPs to operate safely and traceably:

  - Serena: `mcp_serena_activate_project`, `mcp_serena_find_file`, `mcp_serena_search_for_pattern`, `mcp_serena_insert_after_symbol`, `mcp_serena_replace_symbol_body`, `mcp_serena_get_symbols_overview`, `mcp_serena_find_referencing_symbols`.
  - GitHub: `mcp_github-mcp-se_create_branch`, `mcp_github-mcp-se_push_files`, `mcp_github-mcp-se_create_pull_request`, `mcp_github-mcp-se_request_copilot_review`, `mcp_github-mcp-se_add_comment_to_pending_review`.

  **Concrete steps to automate locally (script-friendly):**

  1. `uv sync` (ensure env)
  2. `git checkout -b feat/benchmarks-specs-ci`
  3. Implement `AbstractOptimizer.benchmark(...)` and `AbstractMultiObjectiveOptimizer.benchmark(...)` with `store=True` behavior and `export_benchmark_json` helper.
  4. Add `tests/benchmark_quick/test_[algorithm]_benchmark.py` with a small dim/max_iter and an assertion that the produced JSON validates against `docs/schemas/benchmark-data-schema.json`.
  5. Add GitHub Actions workflow `ci/benchmarks-quick.yml` that runs `uv run pytest -q -k benchmark_quick` on PRs, and `ci/benchmarks-full.yml` scheduled nightly.
  6. Update `.pre-commit-config.yaml` to include a `validate-benchmark-json` hook that runs `python scripts/validate_benchmark_json.py` against generated artifacts.
  7. Run `uv run pytest -q -k benchmark_quick` locally and ensure tests pass, then `git add -A && git commit -m "feat(benchmarks): add benchmark export and quick checks" && git push origin feat/benchmarks-specs-ci`.

  Use `mcp_ai-agent-guid_gap-frameworks-analyzers` to prepare a short gap report for reviewers and `mcp_ai-agent-guid_guidelines-validator` to ensure docstrings and schema usage are compliant with repository standards.

  Use `mcp_serena_think_about_task_adherence` and `mcp_serena_think_about_whether_you_are_done` before opening the PR. Use `mcp_github-mcp-se_add_comment_to_pending_review` to leave guidance in reviews on how to validate generated artifacts.

## Script-first automation & MCP usage

Ensure all operational steps are implemented as scripts (not as runtime network calls embedded in library code) and are enforceable by pre-commit hooks and CI. The following rules are mandatory and must be followed from first principle:

- **Script-first**: All automation must be implemented as scripts under the `scripts/` directory (e.g., `scripts/validate_benchmark_json.py`, `scripts/check_forbidden_mcp_calls.py`). Scripts are invoked by pre-commit hooks and CI workflows; do not add runtime agent or network calls inside `opt/` modules.
- **Pre-commit enforcement**: Add pre-commit hooks for any policy that affects repository hygiene (JSON schema validation, forbidden MCP usage, trivial doctest detection). Pre-commit is the _first_ gate for automation and must be used for local enforcement.
- **No MCP calls in library code**: Never place `mcp_*` calls or agent orchestration inside the `opt/` package or any runtime library modules. MCP usage is part of the development workflow (spec, plan, CI orchestration) and should be executed by scripts or CI jobs, not at runtime in optimizer code.
- **MCP call signatures**: When referencing an MCP in prompts/specs, always include the MCP `id` and a fully-specified parameter object (use the exact parameter names). Example:

  - `mcp_context7_get-library-docs({ context7CompatibleLibraryID: '/github/spec-kit', mode: 'code', topic: 'spec-template' })`
  - `mcp_ai-agent-guid_clarify({ taskDescription: 'Replace doctests with benchmark tests', maxQuestions: 5 })`

- **Recorded MCP usage**: Every speckit action that uses an MCP must log: `mcp: <id> params: <json>` to the Serena memory (use `mcp_serena_write_memory`) so operations remain auditable and reproducible.

- **Forbidden patterns check**: The following patterns are forbidden in `opt/` files and will fail pre-commit: `mcp_`, `mcpContext7`, `agent.run`, `openai.`. Use `scripts/check_forbidden_mcp_calls.py` to detect these patterns and fail fast.

- **MCP List reference**: Use the canonical MCP list (the JSON attached to the spec) and always include the example call from the list when writing prompts. When in doubt, call `mcp_context7_resolve-library-id` first to locate the library and `mcp_context7_get-library-docs` to fetch authoritative templates.

---

## Additional commands for enhanced quality and validation:

### Command Description

- /speckit.clarify Clarify underspecified areas (recommended before /speckit.plan; formerly /quizme)

  Answer: Use `mcp_ai-agent-guid_clarify` to generate a focused set of targeted clarification questions (max 5) for any given spec area. The output MUST be a small JSON object with keys: `questions: list[str]`, `rationale: str`, and `priority: str` (high/medium/low). Use the following prompt template with the MCP verbatim:

  "Given the feature: 'Replace trivial doctests with spec-driven benchmark tests that export schema-compliant JSON', list up to 5 targeted clarification questions, explain why each is necessary (rationale), and assign a priority to each question. Please return a JSON object with keys `questions`, `rationale`, and `priority` so it can be programmatically ingested."

  Required MCPs & usage:

  - `mcp_ai-agent-guid_clarify` (primary)
  - `mcp_context7_get-library-docs` (for reference templates if needed)
  - `mcp_serena_write_memory` (store clarifications as `/memories/benchmarks-clarifications.md`)

  Expected output: JSON with 3-5 questions and rationales. Local verification: check a new memory file exists (`/memories/benchmarks-clarifications.md`) and contains the JSON.

- /speckit.analyze Cross-artifact consistency & coverage analysis (run after /speckit.tasks, before /speckit.implement)

  Answer: Run `mcp_ai-agent-guid_gap-frameworks-analyzers` with the spec content and repository context to produce a prioritized gap report that maps missing items to concrete tasks. The analysis MUST include: `category` (testing/perf/docs/process), `gaps` (detailed), `impact` (low/med/high), and `actionable_tasks` (short task strings). Use this prompt shell:

  "Analyze the spec for replacing doctests with benchmark-driven tests in the 'useful-optimizer' repo. Check coverage of: specs, tests, CI jobs, JSON schema validation, and artifact archival. Produce a JSON report with fields: category, gaps, impact, actionable_tasks. Prioritize tasks by impact and effort."

  Required MCPs & usage:

  - `mcp_ai-agent-guid_gap-frameworks-analyzers` (primary)
  - `mcp_ai-agent-guid_iterative-coverage-enhancer` (to identify test gaps)
  - `mcp_serena_search_for_pattern` and `mcp_serena_find_file` (to locate relevant files, tests, and doctests in repo)

  Expected output: JSON report + prioritized task list. Local verification: a short markdown summary saved to `.github/analysis/benchmarks-gap-report.md` and the actionable tasks added to the speckit plan.

- /speckit.checklist Generate custom quality checklists that validate requirements completeness, clarity, and consistency (like "unit tests for English")

  Answer: Use `mcp_ai-agent-guid_quick-developer-prompts-builder` together with `mcp_ai-agent-guid_guidelines-validator` to generate a reviewer checklist in markdown format, saved at `.github/checklists/benchmarks-spec.md`. The checklist must contain explicit pass/fail checks and commands to run locally, for example:

  - [ ] Spec exists for each removed doctest (`grep -q "benchmark(store=True)" opt -R`)
  - [ ] `benchmark_quick` tests pass: `uv run pytest -q -k benchmark_quick`
  - [ ] Generated JSON validates with `python scripts/validate_benchmark_json.py --schema docs/schemas/benchmark-data-schema.json benchmarks/output/*.json`
  - [ ] Pre-commit hook `validate-benchmark-json` passes on generated artifacts
  - [ ] CI workflows added for quick PR checks and nightly archival

  Required MCPs & usage:

  - `mcp_ai-agent-guid_quick-developer-prompts-builder` (generate suggestions)
  - `mcp_ai-agent-guid_guidelines-validator` (validate checklist items' language and clarity)
  - `mcp_serena_insert_after_symbol` (to insert checklist references into the PR template or contributing docs)

  Expected output: `.github/checklists/benchmarks-spec.md` (markdown checklist). Local verification: run the listed commands and ensure they succeed; include a small CI smoke job that runs the checklist on PRs.
