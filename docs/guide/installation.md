# Installation

## Requirements

- **Python**: 3.10 - 3.12
- **Dependencies**: numpy, scipy, scikit-learn

## Quick Install

::: code-group

```bash [pip]
pip install git+https://github.com/Anselmoo/useful-optimizer
```

```bash [uv (recommended)]
uv add git+https://github.com/Anselmoo/useful-optimizer
```

```bash [poetry]
poetry add git+https://github.com/Anselmoo/useful-optimizer
```

:::

## Development Installation

For contributing or local development:

```bash
# Clone the repository
git clone https://github.com/Anselmoo/useful-optimizer.git
cd useful-optimizer

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Optional Dependencies

### Visualization Support

For benchmark visualization capabilities:

```bash
# With pip
pip install git+https://github.com/Anselmoo/useful-optimizer[visualization]

# With uv
uv add git+https://github.com/Anselmoo/useful-optimizer --extra visualization
```

### Benchmark Support

For running benchmark suites with advanced plotting:

```bash
# With pip
pip install git+https://github.com/Anselmoo/useful-optimizer[benchmark]

# With uv
uv add git+https://github.com/Anselmoo/useful-optimizer --extra benchmark
```

## Verifying Installation

After installation, verify everything works:

```python
# Basic import test
from opt import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

# Quick optimization test
optimizer = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-2.768,
    upper_bound=2.768,
    dim=2,
    max_iter=50
)

best_solution, best_fitness = optimizer.search()
print(f"Installation verified! Best fitness: {best_fitness:.6f}")
```

## Troubleshooting

### ImportError: No module named 'opt'

Ensure the package is installed in your active Python environment:

```bash
pip list | grep useful-optimizer
```

### NumPy Version Conflicts

If you encounter NumPy compatibility issues:

```bash
pip install numpy>=1.26.4
```

### SciPy Missing

Some classical optimizers require SciPy:

```bash
pip install scipy>=1.12.0
```
