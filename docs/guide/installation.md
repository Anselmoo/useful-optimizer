# Installation

This guide covers the installation of Useful Optimizer and its dependencies.

## Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Operating System**: Linux, macOS, or Windows

## Installation Methods

=== "pip"

    Install from GitHub using pip:

    ```bash
    pip install git+https://github.com/Anselmoo/useful-optimizer
    ```

=== "uv (recommended)"

    Install using uv for faster dependency resolution:

    ```bash
    uv add git+https://github.com/Anselmoo/useful-optimizer
    ```

=== "From source"

    Clone the repository and install in development mode:

    ```bash
    git clone https://github.com/Anselmoo/useful-optimizer.git
    cd useful-optimizer
    pip install -e .
    ```

## Optional Dependencies

Useful Optimizer has several optional dependency groups:

### Development Dependencies

For contributing and testing:

```bash
pip install "useful-optimizer[dev]"
```

Includes: `ruff`, `pytest`, `pre-commit`

### Visualization Dependencies

For plotting and visualization:

```bash
pip install "useful-optimizer[visualization]"
```

Includes: `matplotlib`

### Benchmark Dependencies

For running benchmarks and comparisons:

```bash
pip install "useful-optimizer[benchmark]"
```

Includes: `matplotlib`, `seaborn`

### Documentation Dependencies

For building the documentation locally:

```bash
pip install "useful-optimizer[docs]"
```

Includes: `zensical`, `mkdocstrings[python]`, `mkdocs-material`

## Verifying Installation

After installation, verify everything is working:

```python
import opt
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

print("Import successful!")

# Quick test
pso = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-2.768,
    upper_bound=2.768,
    dim=2,
    max_iter=50
)
best_solution, best_fitness = pso.search()
print(f"Test optimization completed with fitness: {best_fitness:.6f}")
```

## Troubleshooting

### Common Issues

!!! question "ImportError: No module named 'opt'"

    Make sure the package is properly installed:
    ```bash
    pip list | grep useful-optimizer
    ```
    If not listed, reinstall the package.

!!! question "NumPy version conflicts"

    Useful Optimizer requires NumPy >= 1.26.4. If you encounter issues:
    ```bash
    pip install --upgrade numpy>=1.26.4
    ```

!!! question "SciPy import errors"

    Some algorithms depend on SciPy. Ensure it's installed:
    ```bash
    pip install scipy>=1.12.0
    ```

## Next Steps

Now that you have Useful Optimizer installed, check out the [Quick Start Guide](quickstart.md) to run your first optimization.
