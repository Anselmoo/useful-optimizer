## Essential commands for the Spec-Driven Development workflow:

- /speckit.constitution Create or update project governing principles and development guidelines

  Answer: Use the Spec Kit constitution template (fetch via Context7). Prompt: "Draft a project constitution emphasizing: Test-First (TDD), Spec-Driven Development, reproducible benchmarks, CI-enforced spec checks, and memory-efficient history tracking (PR #119)." Tools: `mcp_context7_resolve-library-id` + `mcp_context7_get-library-docs` (library `/github/spec-kit`) and `mcp_ai-agent-guid_guidelines-validator` to validate the constitution.

- /speckit.specify Define what you want to build (requirements and user stories)

  Answer: Use the Spec Kit `spec-template` (resolve `/github/spec-kit`) to write feature specs replacing trivial doctests. Prompt: "Write a Feature Specification for updating doctest examples with spec-driven benchmark tests: include Given/When/Then scenarios, acceptance criteria (reproducible with seed, numeric thresholds), fast sanity tests (<1s) and nightly full-run tests. Include edge cases and 'How to validate locally'." Tools: `mcp_context7_get-library-docs`, `mcp_ai-agent-guid_prompt-chaining-builder` to generate spec drafts.

- /speckit.plan Create technical implementation plans with your chosen tech stack

  Answer: Use `mcp_ai-agent-guid_architecture-design-prompt-builder` to produce a 3-phase implementation plan (design → implement → validate). Include modules: benchmark runner, memory-efficient history store, test harness, CI (quick PR checks + scheduled full-run), Spec Kit checks, docs generation, and performance/observability goals. Also include 'one-big-PR' workflow and rollback plan.

- /speckit.tasks Generate actionable task lists for implementation

  Answer: Use `mcp_ai-agent-guid_prompt-chaining-builder` to expand the plan into prioritized, dependency-ordered tasks (spec creation, tests, CI, docs, validation). Optionally use `mcp_ai-agent-guid_sprint-timeline-calculator` to sequence tasks for the single big PR.

- /speckit.implement Execute all tasks to build the feature according to the plan

  Answer: Use the prompt chain from `mcp_ai-agent-guid_prompt-chaining-builder` to implement: create branch, write specs, add `benchmark_quick` tests, add `@pytest.mark.benchmark_full` tests, validate history tracking (PR #119), add CI workflows, and prepare a single comprehensive PR. Local dev steps: `uv sync`, run quick tests, lint/format, run full integration locally if needed, then push branch and open PR (one big update).

## Additional commands for enhanced quality and validation:

### Command Description

- /speckit.clarify Clarify underspecified areas (recommended before /speckit.plan; formerly /quizme)

  Answer: Use `mcp_ai-agent-guid_clarify` (or ask `speckit.clarify`) to generate targeted clarification questions about scope (which doctests to replace, thresholds for acceptance, CI budget for nightly runs) and to capture answers as spec inputs. Example prompt: "List up to 5 targeted questions to finalize acceptance criteria for benchmark tests replacing doctests."

- /speckit.analyze Cross-artifact consistency & coverage analysis (run after /speckit.tasks, before /speckit.implement)

  Answer: Run `mcp_ai-agent-guid_gap-frameworks-analyzers` focused on testing, performance, process, and maturity. Provide a prioritized gap list and concrete acceptance criteria (e.g., `benchmark_quick` <1s, deterministic with seed, nightly job artifacts preserved). Use `mcp_context7_get-library-docs` to ensure specs match Spec Kit templates.

- /speckit.checklist Generate custom quality checklists that validate requirements completeness, clarity, and consistency (like "unit tests for English")

  Answer: Use `mcp_ai-agent-guid_quick-developer-prompts-builder` or `mcp_ai-agent-guid_checklist` to produce a reviewers' checklist that enforces: Spec exists for each removed doctest, `benchmark_quick` tests present and pass, reproducibility verified (seeds documented), memory-efficient history validated, CI jobs added, and `specify check` included in CI. Include short commands for local validation (e.g., `uv run pytest -q -k benchmark_quick`, `specify check`).
