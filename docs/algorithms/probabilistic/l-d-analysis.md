# L D Analysis

<span class="badge badge-probabilistic">Probabilistic</span>

Linear Discriminant Analysis (LDA).

## Algorithm Overview

This module implements the Linear Discriminant Analysis (LDA). LDA is a method used in
statistics, pattern recognition, and machine learning to find a linear combination of
features that characterizes or separates two or more classes of objects or events.
The resulting combination may be used as a linear classifier, or, more commonly, for
dimensionality reduction before later classification.

LDA is closely related to analysis of variance (ANOVA) and regression analysis, which
also attempt to express one dependent variable as a linear combination of other
features or measurements. However, ANOVA uses categorical independent variables and a
continuous dependent variable, whereas discriminant analysis has continuous independent
variables and a categorical dependent variable (i.e., the class label).

## Usage

```python
from opt.probabilistic.linear_discriminant_analysis import LDAnalysis
from opt.benchmark.functions import sphere

optimizer = LDAnalysis(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
    population_size=50,
)

best_solution, best_fitness = optimizer.search()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness:.6e}")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | Required | Objective function to minimize. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `dim` | `int` | Required | Problem dimensionality. |
| `population_size` | `int` | `100` | Number of samples for LDA training. |
| `max_iter` | `int` | `1000` | Maximum optimization iterations. |
| `number_of_labels` | `int` | `20` | Number of discretization bins for fitness values. |
| `unique_classes` | `int` | `2` | Minimum number of unique classes required for LDA. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Probabilistic Algorithms](/algorithms/probabilistic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`linear_discriminant_analysis.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/probabilistic/linear_discriminant_analysis.py)
:::
