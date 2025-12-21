# Useful Optimizer - GitHub Copilot Instructions

Useful Optimizer is a Python optimization library containing 58 optimization algorithms for numeric problems. The project uses **uv** for dependency management, virtual environments, and packaging while providing a comprehensive collection of metaheuristic, gradient-based, and nature-inspired optimization techniques.

**Always reference these instructions first and fall back to search or shell commands only when you encounter unexpected information that does not match the info here.** All commands are written as **single-line Fish-shell commands** so they can be copied directly into the terminal.

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
uv run python -c "from opt.benchmark.functions import shifted_ackley, rosenbrock, sphere; from opt.particle_swarm import ParticleSwarm; print('Testing basic import and optimizer creation...'); pso = ParticleSwarm(func=shifted_ackley, lower_bound=-2.768, upper_bound=2.768, dim=2, max_iter=50); best_solution, best_fitness = pso.search(); print(f'PSO completed successfully. Fitness: {best_fitness:.6f}'); print('Basic functionality test PASSED')"
```
**Expected: Completes in < 1 second, prints fitness value around 0.001-1.0**

### Scenario 2: Multiple Optimizer Test
```fish
uv run python -c "from opt.benchmark.functions import shifted_ackley, rosenbrock; from opt.harmony_search import HarmonySearch; from opt.ant_colony import AntColony; print('Testing multiple optimizers...'); hs = HarmonySearch(func=rosenbrock, lower_bound=-5, upper_bound=5, dim=2, max_iter=50); _, fitness1 = hs.search(); ac = AntColony(func=shifted_ackley, lower_bound=-2.768, upper_bound=2.768, dim=2, max_iter=50); _, fitness2 = ac.search(); print(f'HS fitness: {fitness1:.6f}, ACO fitness: {fitness2:.6f}'); print('Multiple optimizer test PASSED')"
```
**Expected: Completes in < 1 second, prints two fitness values**

### Scenario 3: Direct Script Execution Test
```fish
uv run python opt/harmony_search.py
```
**Expected: Prints "Best solution found:" and "Best fitness found:" with numerical values**

### Scenario 4: Advanced Import Test
```fish
uv run python -c "from opt.abstract_optimizer import AbstractOptimizer; from opt.benchmark.functions import shifted_ackley, sphere, rosenbrock, ackley; from opt.sgd_momentum import SGDMomentum; from opt.adamw import AdamW; print('Advanced imports successful - gradient-based and base classes work')"
```

## Project Structure

### Key Directories and Files
- `opt/` - Main optimization algorithms (58 Python files)
- `opt/abstract_optimizer.py` - Base class that all optimizers inherit from
- `opt/benchmark/functions.py` - Benchmark functions (shifted_ackley, sphere, rosenbrock, etc.)
- `pyproject.toml` - Project configuration, dependencies, and ruff linting rules
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `.github/workflows/python-publish.yaml` - CI/CD for PyPI publishing

### Common Optimization Algorithms Available
**Nature-Inspired:**
- `ParticleSwarm` - Particle Swarm Optimization
- `HarmonySearch` - Music-inspired metaheuristic
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
  ```
  ```fish
  uv run ruff format opt/
  ```
- The project uses extensive ruff rules - expect to find existing linting issues
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
