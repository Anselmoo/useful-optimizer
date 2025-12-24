---
agent: 'agent'
model: Auto (copilot)
tools: ['read', 'edit', 'search', 'web', 'ai-agent-guidelines/code-analysis-prompt-builder', 'ai-agent-guidelines/digital-enterprise-architect-prompt-builder', 'ai-agent-guidelines/documentation-generator-prompt-builder', 'ai-agent-guidelines/guidelines-validator', 'ai-agent-guidelines/hierarchical-prompt-builder', 'ai-agent-guidelines/hierarchy-level-selector', 'ai-agent-guidelines/l9-distinguished-engineer-prompt-builder', 'ai-agent-guidelines/semantic-code-analyzer', 'ai-agent-guidelines/strategy-frameworks-builder', 'context7/*', 'github/*', 'serena/*', 'agent', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-vscode.vscode-websearchforcopilot/websearch', 'todo']
description: 'Complete VitePress documentation with JSON-driven benchmark visualization'
---

# useful-optimizer VitePress Documentation Completion

## Context

Repository: `Anselmoo/useful-optimizer`
Branch: `docs/algorithm-pages-and-sidebar`
Stack: VitePress v1.6.4, Vue 3, ECharts, Python 3.10+

## Objectives

1. Fix benchmark data collection bugs
2. Register Vue-ECharts components
3. Create missing navigation pages
4. Generate API module documentation
5. Document benchmark functions
6. Implement JSON export pipeline

## Non-Goals

- Generate PNG artifacts
- Modify optimizer algorithm logic
- Restructure existing 117 algorithm pages

## Docstring Formatting Requirements (CRITICAL)

**All docstrings must follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).**

### Parameter Description Format

**CRITICAL**: Parameter descriptions MUST start on the **same line** as the parameter name. This is the official Google style.

**❌ WRONG** (line break after parameter name - common in agent mode):
```python
func (Callable[[ndarray], float]):
    Objective function to minimize. Must accept numpy array and return scalar.
    BBOB functions available in `opt.benchmark.functions`.
```

**✅ CORRECT** (description on same line):
```python
func (Callable[[ndarray], float]): Objective function to minimize. Must accept numpy array and return scalar. BBOB functions available in `opt.benchmark.functions`.
```

**✅ CORRECT** (multi-line with proper indentation):
```python
parameter2 (str): This is a longer definition. I need to include so much
    information that it needs a second line. Notice the indentation.
```

### Key Rules

1. **No line breaks between parameter name and description**
2. **Consistent 4-space indentation**
3. **LaTeX for mathematical symbols**: `$\times$` not `×`, `$\alpha$` not `α`
4. **Budget expressions**: `dim $\times$ 10000` NOT `dim×10000` or `dim*10000`
5. **Dimension notation**: Use plain `dim` (NOT `\text{dim}`) for consistency
6. **Run validation**: `pre-commit run -a` before committing

### References

- [Google Python Style Guide - Functions and Methods](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Sphinx Napoleon - Google Style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

---

## Workflow

### Step 1: Checkout Branch

```bash
cd /Users/hahn/LocalDocuments/GitHub_Forks/useful-optimizer
git checkout docs/algorithm-pages-and-sidebar
```

### Step 2: Fix Benchmark Attribute Bug

**File**: `benchmarks/run_benchmark_suite.py`
**Line**: 148

```python
# FIND:
if hasattr(optimizer, "best_fitness_history"):
    convergence_history = optimizer.best_fitness_history

# REPLACE WITH:
if optimizer.track_history and optimizer.history.get("best_fitness"):
    convergence_history = optimizer.history["best_fitness"]
```

**Validation**: `uv run python -c "from benchmarks.run_benchmark_suite import run_single_benchmark; print('OK')"`

### Step 3: Fix Generate Plots Key

**File**: `benchmarks/generate_plots.py`
**Line**: 69

```python
# FIND:
run.get("convergence_history")

# REPLACE WITH:
run.get("history", {}).get("best_fitness", [])
```

### Step 4: Register Vue Components

**File**: `docs/.vitepress/theme/index.ts`

```typescript
// REPLACE ENTIRE FILE WITH:
import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './style.css'

// ECharts
import { VChart } from 'vue-echarts'
import 'echarts'

// Custom components
import ConvergenceChart from './components/ConvergenceChart.vue'
import ECDFChart from './components/ECDFChart.vue'
import ViolinPlot from './components/ViolinPlot.vue'
import FitnessLandscape3D from './components/FitnessLandscape3D.vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {})
  },
  enhanceApp({ app }) {
    app.component('VChart', VChart)
    app.component('ConvergenceChart', ConvergenceChart)
    app.component('ECDFChart', ECDFChart)
    app.component('ViolinPlot', ViolinPlot)
    app.component('FitnessLandscape3D', FitnessLandscape3D)
  }
} satisfies Theme
```

**Validation**: `cd docs && npm run docs:dev` → No SSR errors

### Step 5: Create Changelog Page

**File**: `docs/changelog.md`

```markdown
# Changelog

All notable changes to this project are documented in this file.

## [0.1.2] - 2024-12-22

### Added
- VitePress documentation site with 117 algorithm pages
- Scientific benchmark visualization with ECharts
- Auto-generated sidebar from codebase

### Fixed
- Optimizer convergence issues and numerical stability errors (#43)
- SocialGroupOptimizer convergence tracking (#50)

## [0.1.1] - 2024-11-15

### Added
- Centralized constants module (#45)
- Automated benchmark visualization system (#46)

## [0.1.0] - 2024-10-01

### Added
- Initial release with 90+ optimization algorithms
- AbstractOptimizer base class with track_history support
- Benchmark functions: sphere, rosenbrock, rastrigin, ackley, griewank
```

### Step 6: Create Contributing Page

**File**: `docs/contributing.md`

```markdown
# Contributing

## Development Setup

```bash
git clone https://github.com/Anselmoo/useful-optimizer.git
cd useful-optimizer
uv sync
```

## Running Tests

```bash
uv run pytest opt/test/ -v
```

## Code Style

```bash
uv run ruff check opt/
uv run ruff format opt/
```

## Adding a New Optimizer

1. Create file in appropriate category: `opt/{category}/{name}.py`
2. Inherit from `AbstractOptimizer`
3. Implement `search()` method returning `tuple[np.ndarray, float]`
4. Add `if self.track_history:` blocks for benchmark support
5. Run `uv run python scripts/generate_docs.py --all --sidebar`

## Documentation

```bash
cd docs
npm install
npm run docs:dev
```

## Pull Request Process

1. Create feature branch from `main`
2. Run linting: `uv run ruff check opt/`
3. Run tests: `uv run pytest`
4. Submit PR with conventional commit message
```

### Step 7: Generate API Module Pages

**Execute for each module**:

| Module | Source | Output |
|--------|--------|--------|
| swarm-intelligence | `opt/swarm_intelligence/__init__.py` | `docs/api/swarm-intelligence.md` |
| evolutionary | `opt/evolutionary/__init__.py` | `docs/api/evolutionary.md` |
| gradient-based | `opt/gradient_based/__init__.py` | `docs/api/gradient-based.md` |
| classical | `opt/classical/__init__.py` | `docs/api/classical.md` |
| metaheuristic | `opt/metaheuristic/__init__.py` | `docs/api/metaheuristic.md` |
| constrained | `opt/constrained/__init__.py` | `docs/api/constrained.md` |
| probabilistic | `opt/probabilistic/__init__.py` | `docs/api/probabilistic.md` |
| benchmark-functions | `opt/benchmark/functions.py` | `docs/api/benchmark-functions.md` |

**Template for each**:

```markdown
# {Module Name}

## Overview

{Description from module docstring}

## Available Classes

| Class | Description |
|-------|-------------|
{For each class in __all__: | `{ClassName}` | {First line of docstring} |}

## Import

```python
from opt.{module_path} import {ClassName}
```

## Example

```python
from opt.{module_path} import {FirstClass}
from opt.benchmark.functions import sphere

optimizer = {FirstClass}(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=100
)

best_solution, best_fitness = optimizer.search()
```
```

### Step 8: Document Missing Benchmark Functions

**File**: `docs/benchmarks/functions.md`
**Append after existing content**:

| Function | Formula | Optimum | Bounds |
|----------|---------|---------|--------|
| `schwefel` | $f(x) = 418.9829n - \sum_{i=1}^{n} x_i \sin(\sqrt{\|x_i\|})$ | $f(420.9687^n) = 0$ | $[-500, 500]^n$ |
| `levi` | $f(x) = \sin^2(3\pi x_1) + (x_1-1)^2(1+\sin^2(3\pi x_2)) + (x_2-1)^2(1+\sin^2(2\pi x_2))$ | $f(1,1) = 0$ | $[-10, 10]^2$ |
| `himmelblau` | $f(x) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2$ | $f(3,2) = 0$ | $[-5, 5]^2$ |
| `eggholder` | $f(x) = -(x_2+47)\sin(\sqrt{\|x_1/2+(x_2+47)\|}) - x_1\sin(\sqrt{\|x_1-(x_2+47)\|})$ | $f(512, 404.2319) = -959.6407$ | $[-512, 512]^2$ |
| `beale` | $f(x) = (1.5-x_1+x_1x_2)^2 + (2.25-x_1+x_1x_2^2)^2 + (2.625-x_1+x_1x_2^3)^2$ | $f(3, 0.5) = 0$ | $[-4.5, 4.5]^2$ |
| `goldstein_price` | See source | $f(0, -1) = 3$ | $[-2, 2]^2$ |
| `booth` | $f(x) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2$ | $f(1, 3) = 0$ | $[-10, 10]^2$ |

### Step 9: Create JSON Export Script

**File**: `benchmarks/export_json.py`

```python
#!/usr/bin/env python3
"""Export benchmark results to VitePress-compatible JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def logarithmic_sample(data: list, max_points: int = 50) -> tuple[list, list]:
    """Sample data at logarithmic intervals."""
    if len(data) <= max_points:
        return list(range(1, len(data) + 1)), data

    indices = np.unique(np.geomspace(1, len(data), max_points).astype(int) - 1)
    return [int(i + 1) for i in indices], [data[i] for i in indices]


def compute_ecdf(values: list[float], thresholds: list[float]) -> list[float]:
    """Compute ECDF proportions for given thresholds."""
    arr = np.array(values)
    return [float(np.mean(arr <= t)) for t in thresholds]


def export_benchmark_json(input_path: Path, output_dir: Path) -> None:
    """Export benchmark results to JSON files."""
    with open(input_path) as f:
        results = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata
    metadata = {
        "generated_at": results["metadata"]["timestamp"],
        "library_version": "0.1.2",
        "dimensions": results["metadata"]["dimensions"],
        "max_iterations": results["metadata"]["max_iterations"],
        "num_runs": results["metadata"]["n_runs"],
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Process each function
    ecdf_thresholds = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100]
    ecdf_data = {"thresholds": ecdf_thresholds, "algorithms": {}}

    for func_name, func_data in results["benchmarks"].items():
        func_output = {"dimensions": {}}

        for dim_key, dim_data in func_data.items():
            func_output["dimensions"][dim_key] = {}

            for algo_name, algo_data in dim_data.items():
                if "runs" not in algo_data:
                    continue

                # Collect convergence data
                all_convergence = []
                final_values = []

                for run in algo_data["runs"]:
                    history = run.get("history", {}).get("best_fitness", [])
                    if history:
                        all_convergence.append(history)
                    final_values.append(run.get("best_fitness", float("inf")))

                # Sample and aggregate
                if all_convergence:
                    max_len = max(len(h) for h in all_convergence)
                    padded = [h + [h[-1]] * (max_len - len(h)) for h in all_convergence]
                    mean_curve = np.mean(padded, axis=0).tolist()
                    std_curve = np.std(padded, axis=0).tolist()

                    evals, sampled_mean = logarithmic_sample(mean_curve)
                    _, sampled_std = logarithmic_sample(std_curve)
                else:
                    evals, sampled_mean, sampled_std = [], [], []

                func_output["dimensions"][dim_key][algo_name] = {
                    "statistics": {
                        "mean_fitness": float(np.mean(final_values)),
                        "std_fitness": float(np.std(final_values)),
                        "min_fitness": float(np.min(final_values)),
                    },
                    "convergence": {
                        "evaluations": evals,
                        "mean_best_fitness": sampled_mean,
                        "std_best_fitness": sampled_std,
                    },
                }

                # Aggregate ECDF
                if algo_name not in ecdf_data["algorithms"]:
                    ecdf_data["algorithms"][algo_name] = []
                ecdf_data["algorithms"][algo_name].extend(final_values)

        with open(output_dir / f"{func_name}.json", "w") as f:
            json.dump(func_output, f, indent=2)

    # Compute final ECDF
    for algo_name in ecdf_data["algorithms"]:
        ecdf_data["algorithms"][algo_name] = compute_ecdf(
            ecdf_data["algorithms"][algo_name], ecdf_thresholds
        )

    with open(output_dir / "ecdf_summary.json", "w") as f:
        json.dump(ecdf_data, f, indent=2)

    print(f"Exported to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmarks/output/results.json")
    parser.add_argument("--output", default="docs/public/benchmarks")
    args = parser.parse_args()

    export_benchmark_json(Path(args.input), Path(args.output))
```

### Step 10: Update Results Page

**File**: `docs/benchmarks/results.md`
**Replace lines 5-55** (mock data) with:

```vue
<script setup>
import { ref, onMounted } from 'vue'

const sphereData = ref(null)
const ecdfData = ref(null)

onMounted(async () => {
  sphereData.value = await fetch('/useful-optimizer/benchmarks/sphere.json').then(r => r.json())
  ecdfData.value = await fetch('/useful-optimizer/benchmarks/ecdf_summary.json').then(r => r.json())
})
</script>
```

### Step 11: Build and Validate

```bash
cd docs
npm run docs:build
```

**Expected**: Build completes with no dead link warnings.

### Step 12: Commit and Push

```bash
git add -A
git commit -m "feat(docs): complete documentation with benchmark visualization"
git push
```

---

## Acceptance Tests

| Test | Command | Expected |
|------|---------|----------|
| Build | `npm run docs:build` | Exit 0, no dead links |
| Changelog | Navigate `/changelog` | Page renders |
| Contributing | Navigate `/contributing` | Page renders |
| API Pages | Navigate `/api/swarm-intelligence` | Page renders with class list |
| Charts | Navigate `/benchmarks/results` | ECharts render |
| JSON Size | `du -sh docs/public/benchmarks/` | < 3MB |

## Edge Cases

| Case | Handling |
|------|----------|
| Empty convergence history | Skip algorithm, log warning |
| Missing `__all__` in module | Use `dir()` to enumerate |
| Function without docstring | Generate placeholder with TODO |

## References

- COCO Platform: Hansen et al., IEEE Trans. Evol. Computation 2022
- Vue-ECharts: https://github.com/ecomfe/vue-echarts
- VitePress: https://vitepress.dev
