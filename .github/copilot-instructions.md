# Useful Optimizer - GitHub Copilot Instructions

Useful Optimizer is a Python optimization library containing **120 optimization algorithms** organized into 10 categories for numeric problems. The project uses **uv** for dependency management, virtual environments, and packaging while providing a comprehensive collection of metaheuristic, gradient-based, and nature-inspired optimization techniques.

**Always reference these instructions first and fall back to search or shell commands only when you encounter unexpected information that does not match the info here.**

## Shell Command Syntax
- **Local terminal commands**: Fish shell (single-line, can be copied directly)
- **Agent/automated mode**: Bash syntax (used in `.github/prompts/*.prompt.md` files)
- All Fish commands use `&&` for chaining, semicolons for inline Python

## Docs Completion Workflow (MANDATORY)
- When working on documentation tasks, **follow `.github/prompts/docs-completion.prompt.md` exactly** (objectives, non-goals, and steps). Do not improvise outside that prompt.
- Use the branch and stack from the prompt: `git checkout docs/algorithm-pages-and-sidebar`, VitePress v1.6.4/Vue 3/ECharts, Python 3.10+.
- Apply these exact code fixes from the prompt verbatim before other changes:

  **Fix 1: `benchmarks/run_benchmark_suite.py` line 148**
  ```python
  # FIND:
  if hasattr(optimizer, "best_fitness_history"):
      convergence_history = optimizer.best_fitness_history

  # REPLACE WITH:
  if optimizer.track_history and optimizer.history.get("best_fitness"):
      convergence_history = optimizer.history["best_fitness"]
  ```

  **Fix 2: `benchmarks/generate_plots.py` line 69**
  ```python
  # FIND:
  run.get("convergence_history")

  # REPLACE WITH:
  run.get("history", {}).get("best_fitness", [])
  ```

  **Fix 3: Register Vue-ECharts in `docs/.vitepress/theme/index.ts`**
  - Replace entire file with the TypeScript code from Step 4 of the prompt
  - Registers: VChart, ConvergenceChart, ECDFChart, ViolinPlot, FitnessLandscape3D

- Generate docs pages and JSON exports as directed; **do not modify optimizer algorithm logic** (explicit non-goal).
- Validation commands (Fish shell for local testing):
  ```fish
  uv run python -c "from benchmarks.run_benchmark_suite import run_single_benchmark; print('OK')"
  cd docs; npm run docs:dev
  ```

## Working Effectively

### Initial Setup and Installation
- Install uv (recommended via installer script):
  ```fish
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Ensure `$HOME/.local/bin` is on your Fish PATH (replace with your actual install dir if different):
  ```fish
  set -Ux PATH $HOME/.local/bin $PATH
  ```
- Install project dependencies and create the managed virtual environment:
  ```fish
  uv sync
  ```
  **NEVER CANCEL: Takes 2-3 minutes to complete. Set timeout to 5+ minutes.**

### Core Development Commands
- Run any Python command in the project environment:
  ```fish
  uv run python [your_command]
  ```
- Test basic functionality:
  ```fish
  uv run python -c "import opt; print('Import successful')"
  ```
- Run linting (finds code quality issues):
  ```fish
  uv run ruff check opt/
  ```
  **Takes 5-10 seconds. Common to find existing issues in codebase.**
- Auto-format code:
  ```fish
  uv run ruff format opt/
  ```
  **Takes less than 1 second.**
- Build the package:
  ```fish
  uv build
  ```
  **Takes less than 1 second. Creates wheel and sdist in dist/ directory.**

## Validation Scenarios

**ALWAYS test your changes with these scenarios after making modifications:**

### Scenario 1: Basic Import and Functionality Test
```fish
uv run python -c "from opt.benchmark.functions import shifted_ackley, rosenbrock, sphere; from opt.swarm_intelligence.particle_swarm import ParticleSwarm; print('Testing basic import and optimizer creation...'); pso = ParticleSwarm(func=shifted_ackley, lower_bound=-2.768, upper_bound=2.768, dim=2, max_iter=50); best_solution, best_fitness = pso.search(); print(f'PSO completed successfully. Fitness: {best_fitness:.6f}'); print('Basic functionality test PASSED')"
```
**Expected: Completes in < 1 second, prints fitness value around 0.001-1.0**

### Scenario 2: Multiple Optimizer Test
```fish
uv run python -c "from opt.benchmark.functions import shifted_ackley, rosenbrock; from opt.metaheuristic.harmony_search import HarmonySearch; from opt.swarm_intelligence.ant_colony import AntColony; print('Testing multiple optimizers...'); hs = HarmonySearch(func=rosenbrock, lower_bound=-5, upper_bound=5, dim=2, max_iter=50); _, fitness1 = hs.search(); ac = AntColony(func=shifted_ackley, lower_bound=-2.768, upper_bound=2.768, dim=2, max_iter=50); _, fitness2 = ac.search(); print(f'HS fitness: {fitness1:.6f}, ACO fitness: {fitness2:.6f}'); print('Multiple optimizer test PASSED')"
```
**Expected: Completes in < 1 second, prints two fitness values**

### Scenario 3: Direct Script Execution Test
```fish
uv run python opt/metaheuristic/harmony_search.py
```
**Expected: Prints "Best solution found:" and "Best fitness found:" with numerical values**

### Scenario 4: Advanced Import Test
```fish
uv run python -c "from opt.abstract_optimizer import AbstractOptimizer; from opt.benchmark.functions import shifted_ackley, sphere, rosenbrock, ackley; from opt.gradient_based.sgd_momentum import SGDMomentum; from opt.gradient_based.adamw import AdamW; print('Advanced imports successful - gradient-based and base classes work')"
```

## Project Structure

### Key Directories and Files
- `opt/` - Main optimization algorithms organized in 10 category subdirectories
  - `classical/` - 9 classical optimization algorithms
  - `constrained/` - 5 constrained optimization algorithms
  - `evolutionary/` - 6 evolutionary algorithms
  - `gradient_based/` - 11 gradient-based optimizers
  - `metaheuristic/` - 14 metaheuristic algorithms
  - `multi_objective/` - 3 multi-objective optimizers (+ abstract base)
  - `physics_inspired/` - 4 physics-inspired algorithms

**Swarm Intelligence (56 algorithms):**
- `ParticleSwarm` - Particle Swarm Optimization (opt.swarm_intelligence.particle_swarm)
- `AntColony` - Ant Colony Optimization (opt.swarm_intelligence.ant_colony)
- `BatAlgorithm` - Bat-inspired algorithm (opt.swarm_intelligence.bat)
- `FireflyAlgorithm` - Firefly Algorithm (opt.swarm_intelligence.firefly)
- `BeeAlgorithm` - Artificial Bee Colony (opt.swarm_intelligence.bee)

**Gradient-Based (11 algorithms):**
- `SGDMomentum` - Stochastic Gradient Descent with momentum (opt.gradient_based.sgd_momentum)
- `AdamW` - Adam with weight decay (opt.gradient_based.adamw)
- `RMSprop` - RMSprop optimizer (opt.gradient_based.rmsprop)
- `Adam` - Adaptive Moment Estimation (opt.gradient_based.adaptive_moment_estimation)

**Classical (9 algorithms):**
- `SimulatedAnnealing` - Simulated Annealing (opt.classical.simulated_annealing)
- `NelderMead` - Nelder-Mead simplex method (opt.classical.nelder_mead)
- `HillClimbing` - Hill Climbing (opt.classical.hill_climbing)
- `BFGS` - Quasi-Newton method (opt.classical.bfgs)

**Metaheuristic (14 algorithms):**
- `HarmonySearch` - Music-inspired metaheuristic (opt.metaheuristic.harmony_search)
- `SineCosine` - Sine Cosine Algorithm (opt.metaheuristic.sine_cosine_algorithm)

**Note:** All optimizers are organized in category subdirectories. Always use the full import path:
`from opt.category_name.module_name import ClassName`ed metaheuristic
- `AntColony` - Ant Colony Optimization
- `BatAlgorithm` - Bat-inspired algorithm
- `FireflyAlgorithm` - Firefly Algorithm

**Gradient-Based:**
- `SGDMomentum` - Stochastic Gradient Descent with momentum
- `AdamW` - Adam with weight decay
- `RMSprop` - RMSprop optimizer
- `BFGS` - Quasi-Newton method

**Classical:**
- `SimulatedAnnealing` - Simulated Annealing
- `NelderMead` - Nelder-Mead simplex method
- `HillClimbing` - Hill Climbing

### Benchmark Functions Available
- `shifted_ackley` - Non-centered Ackley function (commonly used in examples)
- `sphere` - Simple sphere function
- `rosenbrock` - Rosenbrock function
- `ackley` - Standard Ackley function
- `rastrigin` - Rastrigin function
- And 15+ more test functions

## Development Guidelines

### Code Quality and Linting
- **ALWAYS run linting before committing:**
  ```fish
  uv run ruff check opt/ && uv run ruff format opt/
  ```
- Or run them separately:
  ```fish
  uv run ruff check opt/
  ``` (single-objective)
- Or inherit from `AbstractMultiObjectiveOptimizer` in `opt/multi_objective/abstract_multi_objective.py` (multi-objective)
- Implement the required `search()` method returning `tuple[np.ndarray, float]` (or `tuple[ndarray, ndarray]` for multi-objective)
- Follow existing pattern: see `opt/swarm_intelligence/particle_swarm.py` or `opt/metaheuristic/harmony_search.py` as examples
- Place in appropriate category subdirectory: classical/, swarm_intelligence/, gradient_based/, etc.
- Include a `if __name__ == "__main__":` block for direct testing
- Use benchmark functions from `opt.benchmark.functions` for testing
- **IMPORTANT:** Follow COCO/BBOB docstring template in `.github/prompts/optimizer-docs-template.prompt.md`nting issues
- Ruff configuration is in `pyproject.toml` with Google docstring convention
- Pre-commit hooks are configured but run `ruff` manually to be sure

### Creating New Optimizers
- Inherit from `AbstractOptimizer` class in `opt/abstract_optimizer.py`
- Implement the required `search()` method returning `tuple[np.ndarray, float]`
- Follow existing pattern: see `opt/particle_swarm.py` or `opt/harmony_search.py` as examples
- Include a `if __name__ == "__main__":` block for direct testing
- Use benchmark functions from `opt.benchmark.functions` for testing

### Testing Changes
- Run the test suite:
  ```fish
  uv run pytest opt/test/ -v
  ```
- Run specific test files:
  ```fish
  uv run pytest opt/test/test_benchmarks.py -v
  ```
- Test with multiple benchmark functions: `shifted_ackley`, `rosenbrock`, `sphere`
- Test with different parameter combinations
- Ensure optimizers complete within reasonable time (< 1 second for small max_iter)

### Common Issues to Avoid
- Don't modify `uv.lock` manually - use `uv add`, `uv remove`, or `uv sync` to change dependencies
- Ruff linting will fail on many existing files - focus on new/modified code
- Some algorithms use legacy `np.random.rand` calls - documented in README
- Always use `uv run` (or activate `.venv`) so commands execute inside the project environment

### CRITICAL: Fish Shell and Terminal Rules
**All commands MUST be single-line.** Multiline commands crash the terminal in Fish shell.

**Git commit messages:** Always use single-line format with conventional commits:
```fish
git commit -m "feat: add new optimizer algorithm"
```

**Conventional commit types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

**NEVER use multiline commit messages like this (WILL CRASH):**
```fish
# BAD - DO NOT USE
git commit -m "Title

Multiline body"
```

**For commit messages with title and body**, use multiple `-m` flags on one line:
```fish
git commit -m "Title" -m "Body paragraph 1" -m "Body paragraph 2"
```

**For longer commit messages**, use the editor:
```fish
git commit
```

**Chaining commands:** Use `&&` to chain multiple commands:
```fish
git add . && git commit -m "message" && git push
```

**Python -c commands:** Keep on single line with semicolons:
```fish
uv run python -c "import x; print(x.value); do_something()"
```

## Project Information
- **Version:** 0.1.2
- **Python Requirements:** 3.10-3.12 (currently running 3.12.3)
- **Main Dependencies:** numpy, scipy, scikit-learn
- **Dev Dependencies:** ruff (linting and formatting)
- **Package Name:** useful-optimizer
- **License:** MIT

## Build and Distribution
- Build package: `uv build` (< 1 second)
- Built artifacts appear in `dist/` directory
- CI/CD publishes to PyPI on tag pushes via GitHub Actions
- Test PyPI publishing is also configured

## Performance Expectations
- **uv sync:** 2-3 minutes (NEVER CANCEL)
- **Ruff linting:** 5-10 seconds
- **Ruff formatting:** < 1 second
- **Package build (uv build):** < 1 second
- **Small optimization runs:** < 1 second (max_iter=50-100)
- **Import operations:** Nearly instantaneous

Always verify your changes work by running the validation scenarios above before committing.
