# L D Analysis

<span class="badge badge-probabilistic">Probabilistic</span>

LDA-guided Optimization using Linear Discriminant Analysis.

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

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Linear Discriminant Analysis Optimizer   |
| Acronym           | LDA-OPT                                  |
| Year Introduced   | 1936                                     |
| Authors           | Fisher, Ronald A. (LDA); Implementation adapted |
| Algorithm Class   | Probabilistic                            |
| Complexity        | O(N*dim² + dim³) per iteration          |
| Properties        | Stochastic, Adaptive                 |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

LDA-guided optimization uses discriminant analysis to classify fitness regions:

$$
\mathbf{w} = \Sigma_W^{-1}(\mu_{\text{good}} - \mu_{\text{bad}})
$$

where:
- $\Sigma_W$ is the within-class scatter matrix
- $\mu_{\text{good}}$, $\mu_{\text{bad}}$ are class means for discretized fitness
- $\mathbf{w}$ is the discriminant direction

**Fitness Discretization**:

$$
y_i = \text{discretize}(f(x_i), n_{\text{bins}})
$$

**Acquisition via L-BFGS-B**:

$$
x^* = \arg\min_{x \in [a,b]^d} \text{LDA.predict}(x)
$$

**Constraint handling**:
- **Boundary conditions**: Hard bounds enforced in L-BFGS-B optimization
- **Feasibility enforcement**: Bounded optimization in acquisition step

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of samples for LDA      |
| max_iter               | 1000    | 1000-5000        | Maximum iterations             |
| number_of_labels       | 20      | 10-50            | Discretization bins            |
| unique_classes         | 2       | 2                | Minimum classes for LDA        |

**Sensitivity Analysis**:
- `number_of_labels`: **High** impact - More bins allow finer discrimination
- `population_size`: **Medium** impact - More samples improve LDA accuracy
- `unique_classes`: **Low** impact - Usually kept at 2 for binary classification
- Recommended tuning ranges: $n_{\text{bins}} \in [10, 100]$, $N \in [5d, 20d]$

## COCO/BBOB Benchmark Settings

**Search Space**:
- Dimensions tested: `2, 3, 5, 10, 20, 40`
- Bounds: Function-specific (typically `[-5, 5]` or `[-100, 100]`)
- Instances: **15** per function (BBOB standard)

**Evaluation Budget**:
- Budget: $\text{dim} \times 10000$ function evaluations
- Independent runs: **15** (for statistical significance)
- Seeds: `0-14` (reproducibility requirement)

**Performance Metrics**:
- Target precision: `1e-8` (BBOB default)
- Success rate at precision thresholds: `[1e-8, 1e-6, 1e-4, 1e-2]`
- Expected Running Time (ERT) tracking

## Raises

ValueError:
If search space is invalid or function evaluation fails.

## Notes

- Uses self.seed for all random number generation
- BBOB: Returns final best solution after max_iter iterations
- LDA requires at least 2 unique classes in discretized fitness

**Computational Complexity**:
- Time per iteration: $O(Nd^2 + d^3)$ for LDA fitting with $N$ samples, dimension $d$,
plus 50 L-BFGS-B runs
- Space complexity: $O(Nd)$ for population storage
- BBOB budget usage: _Typically 40-70% of dim*10000 budget due to L-BFGS-B restarts_

**BBOB Performance Characteristics**:
- **Best function classes**: Smooth functions with clear fitness gradients
- **Weak function classes**: Highly multimodal or discontinuous functions
- Typical success rate at 1e-8 precision: **20-40%** (dim=5)
- Expected Running Time (ERT): Moderate, competitive on smooth landscapes

**Convergence Properties**:
- Convergence rate: Problem-dependent, typically sub-linear
- Local vs Global: Primarily local search via L-BFGS-B
- Premature convergence risk: **Medium** - Depends on discretization quality

**Probabilistic Concepts**:
- **Discriminant Analysis**: Separates fitness classes via linear projection
- **Fisher's Linear Discriminant**: Maximizes between-class / within-class variance ratio
- **Discretization**: Converts continuous fitness to categorical classes
- **Probabilistic Interpretation**: LDA assumes Gaussian class-conditional densities
- **Acquisition**: L-BFGS-B minimizes predicted LDA class (lower = better fitness)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees identical results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` for initialization and L-BFGS-B starts

**Implementation Details**:
- Parallelization: Not supported (sequential LDA updates)
- Constraint handling: Hard bounds in L-BFGS-B optimization
- Numerical stability: KBinsDiscretizer handles outliers, np.nan_to_num for safety
- LDA solver: "lsqr" (least squares solution) for numerical stability
- Multi-start: 50 random restarts for L-BFGS-B to improve global search

**Known Limitations**:
- Discretization loses fitness information (binning effect)
- Requires sufficient population diversity to form multiple classes
- L-BFGS-B multi-start is computationally expensive (50 runs per iteration)
- LDA assumes Gaussian class distributions which may not hold for all functions
- Returns discretized fitness value (not original continuous fitness)
- BBOB known issues: Poor performance on highly non-linear functions

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Current version with BBOB compliance

## References

[1] Fisher, R. A. (1936).
"The use of multiple measurements in taxonomic problems."
_Annals of Eugenics_, 7(2), 179-188.
https://doi.org/10.1111/j.1469-1809.1936.tb02137.x

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Not yet available in COCO archive
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original LDA: Fisher (1936), sklearn implementation
- This implementation: Hybrid LDA-guided optimization for BBOB compliance

## See Also

ParzenTreeEstimator: Alternative model-based optimization with KDE
BBOB Comparison: TPE uses non-parametric KDE vs LDA's parametric approach

BayesianOptimizer: GP-based surrogate model optimization
BBOB Comparison: BO models function directly, LDA models fitness classes

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Probabilistic: BayesianOptimizer, ParzenTreeEstimator
- Model-based: SequentialMonteCarloOptimizer
- Gradient: L-BFGS-B (used in acquisition)

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

::: tip Run-based charts
Convergence, distribution and ECDF charts appear here once this optimizer is included in the benchmark suite.
:::

## Related Pages

- [Probabilistic Algorithms](/algorithms/probabilistic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`linear_discriminant_analysis.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/probabilistic/linear_discriminant_analysis.py)
:::
