---
title: Installation
description: How to install Useful Optimizer
---

# Installation

Useful Optimizer can be installed using several methods. Choose the one that best fits your workflow.

---

## Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Dependencies**: numpy, scipy, scikit-learn (installed automatically)

---

## Installation Methods

### Using pip (PyPI)

The simplest way to install Useful Optimizer:

```bash
pip install useful-optimizer
```

### Using uv (Recommended)

[uv](https://astral.sh/uv) is a fast Python package manager. If you're using uv:

```bash
uv add useful-optimizer
```

### From GitHub (Latest Development)

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/Anselmoo/useful-optimizer
```

Or with uv:

```bash
uv add git+https://github.com/Anselmoo/useful-optimizer
```

### From Source

For development or to contribute:

```bash
git clone https://github.com/Anselmoo/useful-optimizer.git
cd useful-optimizer
pip install -e .
```

Or with uv:

```bash
git clone https://github.com/Anselmoo/useful-optimizer.git
cd useful-optimizer
uv sync
```

---

## Optional Dependencies

### Visualization Support

To enable visualization features (matplotlib-based):

```bash
pip install useful-optimizer[visualization]
```

Or:

```bash
uv add useful-optimizer --extra visualization
```

### Development Dependencies

For development and testing:

```bash
pip install useful-optimizer[dev]
```

---

## Verify Installation

After installation, verify everything is working:

```python
# Quick test
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import sphere

pso = ParticleSwarm(func=sphere, lower_bound=-5, upper_bound=5, dim=2, max_iter=50)
solution, fitness = pso.search()
print(f"Test completed! Fitness: {fitness:.6f}")
```

If you see output without errors, you're ready to go! 🎉

---

## Troubleshooting

### Common Issues

!!! warning "Import Error"
    If you get `ModuleNotFoundError: No module named 'opt'`, ensure:

    1. You've installed the package correctly
    2. You're using the correct Python environment
    3. Try reinstalling: `pip uninstall useful-optimizer && pip install useful-optimizer`

!!! warning "NumPy Version Conflicts"
    If you encounter NumPy-related errors, try:

    ```bash
    pip install --upgrade numpy>=1.26.4
    ```

---

## Next Steps

Now that you have Useful Optimizer installed, head to the [Quick Start](quickstart.md) guide to run your first optimization!
