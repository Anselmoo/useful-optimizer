# Visualization Module

Comprehensive visualization and stability testing tools for optimization algorithms.

## Features

- **Convergence Curves**: Track best fitness values over iterations
- **Trajectory Plots**: Visualize search paths through 2D solution spaces
- **Average Fitness Tracking**: Monitor population fitness with standard deviation bands
- **Stability Testing**: Run algorithms multiple times with different seeds
- **Statistical Analysis**: Generate box plots, histograms, and summary statistics
- **Multi-Optimizer Comparison**: Compare performance and stability across algorithms

## Installation

The visualization module requires matplotlib as an optional dependency:

```bash
pip install useful-optimizer[visualization]
```

Or install manually:

```bash
pip install matplotlib>=3.7.0
```

## Quick Start

### Basic Visualization

```python
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.benchmark.functions import shifted_ackley
from opt.visualization import Visualizer

# Create optimizer with history tracking enabled
pso = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    max_iter=100,
    track_history=True,  # Enable history tracking
    population_size=30,
)

# Run optimization
best_solution, best_fitness = pso.search()

# Create visualizer and generate plots
viz = Visualizer(pso)
viz.plot_convergence()           # Show convergence curve
viz.plot_trajectory()             # Show 2D search trajectory
viz.plot_average_fitness()        # Show population fitness evolution
viz.plot_all()                    # Generate all plots in one figure
```

### Stability Testing

```python
from opt.visualization import run_stability_test

# Run optimizer multiple times with different seeds
results = run_stability_test(
    optimizer_class=ParticleSwarm,
    func=shifted_ackley,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    max_iter=100,
    seeds=[42, 123, 456, 789, 1011],  # Specific seeds
    # OR use: n_runs=10  # Random seeds
)

# Print statistical summary
results.print_summary()

# Generate visualizations
results.plot_boxplot()
results.plot_histogram()
```

### Compare Multiple Optimizers

```python
from opt.visualization import compare_optimizers_stability
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.evolutionary.genetic_algorithm import GeneticAlgorithm

# Compare two or more optimizers
results_dict, fig = compare_optimizers_stability(
    optimizer_classes=[ParticleSwarm, GeneticAlgorithm],
    func=shifted_ackley,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    max_iter=100,
    n_runs=10,
)

# Access individual results
for name, results in results_dict.items():
    print(f"{name}: {results.summary()}")
```

## API Reference

### Visualizer Class

The `Visualizer` class provides visualization methods for a single optimizer run.

**Constructor:**
```python
Visualizer(optimizer: AbstractOptimizer)
```

**Methods:**

- `plot_convergence(log_scale=False, show=True, ax=None)`: Plot best fitness over iterations
- `plot_trajectory(show=True, ax=None, max_points=1000)`: Plot 2D search trajectory
- `plot_average_fitness(show_std=True, show=True, ax=None)`: Plot population fitness with std bands
- `plot_all(save_path=None)`: Generate comprehensive multi-panel visualization

### StabilityResults Class

Stores and analyzes results from multiple optimizer runs.

**Attributes:**
- `optimizer_name`: Name of the optimizer
- `function_name`: Name of the objective function
- `solutions`: List of best solutions from each run
- `fitness_values`: Array of best fitness values
- `seeds`: List of random seeds used

**Methods:**

- `summary()`: Get statistical summary (mean, std, min, max, median, quartiles)
- `print_summary()`: Print formatted summary
- `plot_boxplot(show=True, save_path=None)`: Generate box plot
- `plot_histogram(bins=20, show=True, save_path=None)`: Generate histogram

### Functions

**run_stability_test()**

Run stability test for an optimization algorithm.

```python
run_stability_test(
    optimizer_class: type[AbstractOptimizer],
    func: Callable,
    lower_bound: float,
    upper_bound: float,
    dim: int,
    max_iter: int = 100,
    seeds: Sequence[int] | None = None,
    n_runs: int = 10,
    verbose: bool = True,
    **optimizer_kwargs
) -> StabilityResults
```

**compare_optimizers_stability()**

Compare stability of multiple optimizers.

```python
compare_optimizers_stability(
    optimizer_classes: list[type[AbstractOptimizer]],
    func: Callable,
    lower_bound: float,
    upper_bound: float,
    dim: int,
    max_iter: int = 100,
    n_runs: int = 10,
    show: bool = True,
    save_path: str | None = None,
) -> tuple[dict[str, StabilityResults], Figure]
```

## History Tracking

To use visualization features, optimizers must be run with `track_history=True`:

```python
optimizer = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    max_iter=100,
    track_history=True,  # Required for visualization
)
```

**Note:** History tracking adds memory overhead proportional to `max_iter × population_size`. For very long runs or large populations, consider using it selectively.

## Advanced Usage

### Custom Matplotlib Integration

The visualization methods return matplotlib Figure objects and accept axes parameters, allowing full customization:

```python
import matplotlib.pyplot as plt

# Create custom layout
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

viz = Visualizer(pso)

# Plot to specific axes
viz.plot_convergence(show=False, ax=axes[0, 0])
viz.plot_trajectory(show=False, ax=axes[0, 1])
viz.plot_average_fitness(show=False, ax=axes[1, 0])
viz.plot_convergence(log_scale=True, show=False, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("custom_visualization.png", dpi=300)
```

### Log Scale Convergence

For functions with wide fitness ranges, use log scale:

```python
viz.plot_convergence(log_scale=True)
```

### Saving Plots

All plot methods support saving:

```python
# Individual plots
viz.plot_convergence(show=False)
plt.savefig("convergence.png", dpi=300, bbox_inches="tight")

# Or use save_path parameter
viz.plot_all(save_path="all_plots.png")
results.plot_boxplot(save_path="stability.png")
```

### Reproducible Results

Use specific seeds for reproducible stability tests:

```python
results = run_stability_test(
    optimizer_class=ParticleSwarm,
    func=shifted_ackley,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    seeds=[42, 123, 456],  # Same seeds = same results
)
```

## Examples

See `examples_visualization.py` for complete working examples including:

1. Basic visualization workflow
2. Stability testing with multiple seeds
3. Multi-optimizer comparison
4. Log scale convergence plots
5. Custom matplotlib integration

Run the examples:

```bash
python examples_visualization.py
```

## Supported Optimizers

The visualization module works with **all 58+ optimizers** in the package that inherit from `AbstractOptimizer`. This includes:

- **Swarm Intelligence**: ParticleSwarm, AntColony, FireflyAlgorithm, etc.
- **Evolutionary**: GeneticAlgorithm, DifferentialEvolution, CMAESAlgorithm, etc.
- **Gradient-Based**: AdamW, SGDMomentum, BFGS, etc.
- **Metaheuristic**: SimulatedAnnealing, TabuSearch, HarmonySearch, etc.
- And many more!

## Performance Considerations

- **History Tracking**: Adds `O(max_iter × population_size)` memory overhead
- **2D Trajectory**: Only available for 2D problems (dim=2)
- **Large Runs**: For `max_iter > 10000`, consider using `max_points` parameter in `plot_trajectory()`
- **Stability Tests**: Running N tests with M iterations each requires `N × M` function evaluations

## Tips

1. **Start Small**: Test with `max_iter=50-100` before running longer optimizations
2. **Use Seeds**: Specify seeds for reproducible results in papers/reports
3. **Compare Fairly**: Use same `max_iter`, bounds, and function for comparison
4. **Check Convergence**: Use log scale to see if optimizer is still improving
5. **Population Diversity**: Use `plot_average_fitness()` to monitor exploration vs exploitation

## Troubleshooting

**ValueError: "track_history=True"**
- Ensure optimizer is created with `track_history=True`

**ValueError: "2D problems"**
- Trajectory plotting only works for `dim=2`
- Other plots work for any dimensionality

**Memory Issues**
- Reduce `max_iter` or `population_size`
- Don't track history for production runs

**Different Results**
- Ensure same `seed` value for reproducibility
- Check that function evaluations are deterministic

## Citation

If you use this visualization module in your research, please cite:

```bibtex
@software{useful_optimizer,
  title = {Useful Optimizer: A Collection of Optimization Algorithms},
  author = {Hahn, Anselm},
  year = {2024},
  url = {https://github.com/Anselmoo/useful-optimizer}
}
```
