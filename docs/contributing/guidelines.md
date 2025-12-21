---
title: Contributing Guidelines
description: How to contribute to Useful Optimizer
---

# Contributing Guidelines

Thank you for your interest in contributing to Useful Optimizer! This guide will help you get started.

---

## Getting Started

### Prerequisites

- Python 3.10, 3.11, or 3.12
- [uv](https://astral.sh/uv) (recommended) or pip
- Git

### Setup Development Environment

1. **Fork and clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/useful-optimizer.git
    cd useful-optimizer
    ```

2. **Install dependencies with uv:**

    ```bash
    uv sync
    ```

3. **Verify installation:**

    ```bash
    uv run python -c "import opt; print('Setup complete!')"
    ```

---

## Development Workflow

### Code Style

We use **Ruff** for linting and formatting:

```bash
# Check for issues
uv run ruff check opt/

# Auto-fix issues
uv run ruff check opt/ --fix

# Format code
uv run ruff format opt/
```

### Running Tests

```bash
# Run all tests
uv run pytest opt/test/ -v

# Run specific test file
uv run pytest opt/test/test_benchmarks.py -v

# Run with coverage
uv run pytest opt/test/ -v --cov=opt
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code before commits:

```bash
uv run pre-commit install
```

---

## Adding a New Algorithm

1. **Create the algorithm file** in the appropriate category folder:

    ```
    opt/
    └── swarm_intelligence/
        └── my_new_algorithm.py
    ```

2. **Follow the template structure:**

    ```python
    """My New Algorithm implementation.

    This module implements the My New Algorithm optimization method.
    """
    from __future__ import annotations

    from typing import TYPE_CHECKING

    import numpy as np

    from opt.abstract_optimizer import AbstractOptimizer

    if TYPE_CHECKING:
        from collections.abc import Callable


    class MyNewAlgorithm(AbstractOptimizer):
        """My New Algorithm optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower boundary of search space.
            upper_bound: Upper boundary of search space.
            dim: Problem dimensionality.
            max_iter: Maximum iterations.
            # Add algorithm-specific parameters...
        """

        def __init__(
            self,
            func: Callable[[np.ndarray], float],
            lower_bound: float,
            upper_bound: float,
            dim: int,
            max_iter: int = 100,
            # Add algorithm-specific parameters...
        ) -> None:
            super().__init__(func, lower_bound, upper_bound, dim, max_iter)
            # Initialize algorithm-specific attributes...

        def search(self) -> tuple[np.ndarray, float]:
            """Execute optimization.

            Returns:
                Tuple of (best_solution, best_fitness).
            """
            # Implement the algorithm...
            return self.best_solution, self.best_fitness


    if __name__ == "__main__":
        from opt.demo import run_demo
        run_demo(MyNewAlgorithm)
    ```

3. **Export from the category `__init__.py`:**

    ```python
    from opt.swarm_intelligence.my_new_algorithm import MyNewAlgorithm

    __all__ = [..., "MyNewAlgorithm"]
    ```

4. **Add tests** in `opt/test/`:

    ```python
    def test_my_new_algorithm():
        from opt.swarm_intelligence import MyNewAlgorithm
        from opt.benchmark.functions import sphere

        optimizer = MyNewAlgorithm(
            func=sphere,
            lower_bound=-5,
            upper_bound=5,
            dim=2,
            max_iter=50,
        )
        solution, fitness = optimizer.search()
        assert fitness < 1.0  # Should find a reasonable solution
    ```

5. **Document the algorithm** (optional but appreciated):

    Create `docs/algorithms/category/my-new-algorithm.md`

---

## Pull Request Process

1. **Create a feature branch:**

    ```bash
    git checkout -b feature/my-new-algorithm
    ```

2. **Make your changes** following the guidelines above

3. **Ensure all checks pass:**

    ```bash
    uv run ruff check opt/
    uv run ruff format opt/
    uv run pytest opt/test/ -v
    ```

4. **Commit with a descriptive message:**

    ```bash
    git commit -m "feat: add My New Algorithm to swarm intelligence"
    ```

5. **Push and create a Pull Request**

---

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `test` | Adding or updating tests |
| `refactor` | Code refactoring |
| `chore` | Maintenance tasks |

Examples:

- `feat: add Whale Optimization Algorithm`
- `fix: correct bounds checking in PSO`
- `docs: add mathematical formulation to ACO page`
- `test: add convergence tests for gradient optimizers`

---

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Give credit where it's due

---

## Questions?

- Open an [issue](https://github.com/Anselmoo/useful-optimizer/issues) for bugs or feature requests
- Start a [discussion](https://github.com/Anselmoo/useful-optimizer/discussions) for questions

Thank you for contributing! 🎉
