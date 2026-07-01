# Bayesian Optimizer

<span class="badge badge-probabilistic">Probabilistic</span>

Bayesian Optimization (BO) using Gaussian Process surrogates.

## Algorithm Overview

This module implements Bayesian Optimization, a probabilistic optimization
technique using Gaussian Process surrogate models.

The algorithm builds a probabilistic model of the objective function and
uses it to select promising points to evaluate.

## Usage

```python
from opt.probabilistic.bayesian_optimizer import BayesianOptimizer
from opt.benchmark.functions import sphere

optimizer = BayesianOptimizer(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
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
| `n_initial` | `int` | `10` | Number of initial random samples to build GP surrogate. |
| `max_iter` | `int` | `50` | Maximum Bayesian optimization iterations after initial
        sampling. |
| `xi` | `float` | `0.01` | Exploration parameter for Expected Improvement acquisition. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Bayesian Optimization                    |
| Acronym           | BO                                       |
| Year Introduced   | 2012                                     |
| Authors           | Snoek, Jasper; Larochelle, Hugo; Adams, Ryan P. |
| Algorithm Class   | Probabilistic                            |
| Complexity        | O(n³) per iteration (GP regression)      |
| Properties        | Stochastic, Adaptive                 |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Bayesian Optimization models the objective function using a Gaussian Process (GP) posterior:

$$
f(x) \sim \mathcal{GP}(\mu(x), k(x, x'))
$$

where:
- $\mu(x)$ is the posterior mean function
- $k(x, x')$ is the covariance kernel (RBF/squared exponential)
- $f(x)$ is the unknown objective function

**Acquisition Function** (Expected Improvement):

$$
\text{EI}(x) = \mathbb{E}[\max(f_{\text{best}} - f(x), 0)]
$$

$$
\text{EI}(x) = (\mu(x) - f_{\text{best}} - \xi)\Phi(Z) + \sigma(x)\phi(Z)
$$

where:
- $\Phi$ is the standard normal CDF
- $\phi$ is the standard normal PDF
- $Z = \frac{\mu(x) - f_{\text{best}} - \xi}{\sigma(x)}$
- $\xi$ is the exploration parameter
- $\sigma(x)$ is the posterior standard deviation

**Constraint handling**:
- **Boundary conditions**: Clamping to bounds during optimization
- **Feasibility enforcement**: Bounds enforced in acquisition function optimization

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| n_initial              | 10      | 2*dim            | Initial random samples         |
| max_iter               | 50      | 100-500          | Maximum BO iterations          |
| xi                     | 0.01    | 0.01-0.1         | Exploration-exploitation param |

**Sensitivity Analysis**:
- `n_initial`: **High** impact - More initial samples improve GP accuracy
- `max_iter`: **Medium** impact - BO converges quickly with good surrogate
- `xi`: **Medium** impact - Balances exploration vs exploitation
- Recommended tuning ranges: $\xi \in [0.001, 0.1]$, $n_{\text{initial}} \in [2d, 5d]$

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

ValueError: If search space is invalid or function evaluation fails.

## Notes

- Uses self.seed for all random number generation
- BBOB: Returns final best solution after max_iter evaluations
- GP regression may fail for ill-conditioned data

**Computational Complexity**:
- Time per iteration: $O(n^3)$ for GP regression with $n$ observations
- Space complexity: $O(n^2)$ for covariance matrix storage
- BBOB budget usage: _Typically 10-30% of dim*10000 budget due to expensive GP updates_

**BBOB Performance Characteristics**:
- **Best function classes**: Smooth unimodal functions (Sphere, Ellipsoid, Rosenbrock)
- **Weak function classes**: High-dimensional multimodal, discontinuous functions
- Typical success rate at 1e-8 precision: **40-60%** (dim=5)
- Expected Running Time (ERT): Competitive on smooth functions, poor on rugged landscapes

**Convergence Properties**:
- Convergence rate: Problem-dependent, typically sub-linear to linear
- Local vs Global: Global search capability via acquisition function
- Premature convergence risk: **Low** - EI balances exploration/exploitation

**Probabilistic Concepts**:
- **Prior**: Gaussian Process with RBF kernel as function prior
- **Likelihood**: Gaussian observation model with noise variance
- **Posterior**: GP posterior updated with observed data $(x_i, f(x_i))$
- **Acquisition**: Expected Improvement quantifies value of evaluating point

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees identical results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported (sequential acquisition)
- Constraint handling: Clamping to bounds in acquisition optimization
- Numerical stability: Cholesky decomposition with fallback to mean/std defaults
- Kernel: RBF (squared exponential) with length_scale=1.0

**Known Limitations**:
- Computational cost scales poorly with evaluation count ($O(n^3)$)
- GP regression may fail for near-duplicate points (add jitter if needed)
- Not suitable for high-dimensional problems (dim > 20)
- BBOB known issues: Slow convergence on ill-conditioned problems

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Current version with BBOB compliance

## References

[1] Snoek, J., Larochelle, H., & Adams, R. P. (2012).
"Practical Bayesian Optimization of Machine Learning Algorithms."
_Advances in Neural Information Processing Systems_ 25 (NIPS 2012).
https://papers.nips.cc/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Not yet available in COCO archive
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Not publicly available
- This implementation: Based on [1] with RBF kernel and EI acquisition

## See Also

SequentialMonteCarloOptimizer: Population-based probabilistic method
BBOB Comparison: SMC more robust on multimodal, BO faster on smooth unimodal

ParzenTreeEstimator: Tree-structured Parzen estimator (TPE) for hyperparameter optimization
BBOB Comparison: TPE similar convergence, less computational cost than BO

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Probabilistic: AdaptiveMetropolisOptimizer, SequentialMonteCarloOptimizer
- Gradient: AdamW, SGDMomentum
- Metaheuristic: SimulatedAnnealing, HarmonySearch

## Related Pages

- [Probabilistic Algorithms](/algorithms/probabilistic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`bayesian_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/probabilistic/bayesian_optimizer.py)
:::
