---
agent: 'agent'
model: Auto (copilot)
tools: ['read', 'edit', 'search', 'web', 'ai-agent-guidelines/code-analysis-prompt-builder', 'ai-agent-guidelines/digital-enterprise-architect-prompt-builder', 'ai-agent-guidelines/documentation-generator-prompt-builder', 'ai-agent-guidelines/guidelines-validator', 'ai-agent-guidelines/hierarchical-prompt-builder', 'ai-agent-guidelines/hierarchy-level-selector', 'ai-agent-guidelines/l9-distinguished-engineer-prompt-builder', 'ai-agent-guidelines/semantic-code-analyzer', 'ai-agent-guidelines/strategy-frameworks-builder', 'context7/*', 'github/*', 'serena/*', 'agent', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-vscode.vscode-websearchforcopilot/websearch', 'todo']
description: 'Generate a comprehensive COCO/BBOB compliant optimizer docstring template for the useful-optimizer library, ensuring all required sections and formatting conventions are included for scientific reproducibility and benchmark compliance.'
---
# COCO/BBOB Compliant Optimizer Docstring Template

This template provides a standardized format for documenting optimization algorithms in the `useful-optimizer` library to ensure COCO/BBOB benchmark compliance and scientific reproducibility.

## Formatting Conventions

When writing docstrings, use the following Markdown formatting to enhance readability:

- **String literal**: Use raw triple double quotes (`r"""..."""`) for optimizer docstrings.
  Raw strings are required because the template includes LaTeX (`$`, `\\`) and code fences.
- **Mathematical expressions**: Use LaTeX notation **EXCLUSIVELY** (Ruff RUF002)
  - Display math: `$$ ... $$` for equations on their own line
  - Inline math: `$...$` for mathematical symbols and variables in text
  - **CRITICAL**: NEVER use unicode symbols (×, α, β, γ, etc.) - ALWAYS use LaTeX:
    - Multiplication: `$\times$` NOT `×`
    - Greek letters: `$\alpha$`, `$\beta$`, `$\gamma$` NOT `α`, `β`, `γ`
    - See: https://docs.astral.sh/ruff/rules/ambiguous-unicode-character-docstring/
- **No line breaks in parameter descriptions**: Keep Args/Attributes on single lines
  - ✅ Correct: `param (type): Description continues on same line. More details here.`
  - ❌ Wrong: Breaking descriptions across multiple lines
- **Accurate indentation**: Use consistent 4-space indentation throughout
- **Emphasis**: Use `**bold**` for important terms, headers, and key values
- **Code elements**: Use `` `code` `` for parameter names, constants, and code snippets
- **Italics**: Use `_italic_` for journal names, emphasis, and notes
- **Lists**: Use `-` for bullet points with proper indentation

These formatting conventions will be parsed when generating documentation.

## Template Structure

All optimizer docstrings **MUST** include the following 11 sections in this exact order:

### 1. Algorithm Metadata Block

Place at the top of the class docstring, formatted as a table:

```python
"""[Algorithm Full Name] ([ACRONYM]) optimization algorithm.

Algorithm Metadata:
    | Property          | Value                                    |
    |-------------------|------------------------------------------|
    | Algorithm Name    | [Full algorithm name]                    |
    | Acronym           | [SHORT]                                  |
    | Year Introduced   | [YYYY]                                   |
    | Authors           | [Last1, First1; Last2, First2]          |
    | Algorithm Class   | [Metaheuristic/Evolutionary/Gradient/...]|
    | Complexity        | O([expression with LaTeX: n $\times$ m]) |
    | Properties        | [Population-based, Derivative-free, ...] |
    | Implementation    | Python 3.10+                             |
    | COCO Compatible   | Yes                                      |
```

### 2. Mathematical Formulation

Provide the core mathematical equations with LaTeX:

```python
Mathematical Formulation:
    Core update equation:

        $$
        x_{t+1} = x_t + v_t
        $$

    where:
        - $x_t$ is the position at iteration $t$
        - $v_t$ is the velocity/step at iteration $t$
        - Additional variable definitions...

    Constraint handling:
        - **Boundary conditions**: [clamping/reflection/periodic]
        - **Feasibility enforcement**: [description]
```

### 3. Hyperparameters

Document all hyperparameters with defaults and BBOB-recommended values:

```python
Hyperparameters:
    | Parameter              | Default | BBOB Recommended | Description                    |
    |------------------------|---------|------------------|--------------------------------|
    | population_size        | 100     | 10*dim           | Number of individuals          |
    | max_iter               | 1000    | 10000            | Maximum iterations             |
    | [param_name]           | [val]   | [bbob_val]       | [description]                  |

    **Sensitivity Analysis**:
        - `[param_name]`: **[High/Medium/Low]** impact on convergence
        - Recommended tuning ranges: $\text{[param]} \in [\text{min}, \text{max}]$
```

### 4. COCO/BBOB Benchmark Settings

Specify standard benchmark configuration:

```python
COCO/BBOB Benchmark Settings:
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
```

### 5. Example

Provide working doctest example with `seed=42`:

```python
Example:
    Basic usage with BBOB benchmark function:

    >>> from opt.[category].[module] import [AlgorithmClass]
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = [AlgorithmClass](
    ...     func=shifted_ackley,
    ...     lower_bound=-2.768,
    ...     upper_bound=2.768,
    ...     dim=2,
    ...     max_iter=100,
    ...     seed=42  # Required for reproducibility
    ... )
    >>> solution, fitness = optimizer.search()
    >>> isinstance(fitness, float) and fitness >= 0
    True

    COCO benchmark example:

    >>> from opt.benchmark.functions import sphere
    >>> optimizer = [AlgorithmClass](
    ...     func=sphere,
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=10,
    ...     max_iter=10,
    ...     population_size=10,
    ...     seed=42,
    ...     track_history=True
    ... )
    >>> solution, fitness = optimizer.search()
    >>> isinstance(fitness, float) and fitness >= 0
    True
    >>> len(optimizer.history.get("best_fitness", [])) > 0
    True
    >>> import tempfile, os, json
    >>> out = tempfile.NamedTemporaryFile(delete=False).name
    >>> from benchmarks import save_run_history
    >>> save_run_history(optimizer, out)
    >>> os.path.exists(out)
    True
```

### 6. Args

Document all parameters with BBOB guidance:

```python
Args:
    func (Callable[[ndarray], float]):
        Objective function to minimize. Must accept numpy array and return scalar.
        BBOB functions available in `opt.benchmark.functions`.
    lower_bound (float):
        Lower bound of search space. BBOB typical: -5 (most functions).
    upper_bound (float):
        Upper bound of search space. BBOB typical: 5 (most functions).
    dim (int):
        Problem dimensionality. BBOB standard dimensions: 2, 3, 5, 10, 20, 40.
    max_iter (int, optional):
        Maximum iterations. BBOB recommendation: 10000 for complete evaluation.
        Defaults to 1000.
    seed (int | None, optional):
        Random seed for reproducibility. BBOB requires seeds 0-14 for 15 runs.
        If None, generates random seed. Defaults to None.
    population_size (int, optional):
        Population size. BBOB recommendation: 10*dim for population-based methods.
        Defaults to 100.
    track_history (bool, optional):
        Enable convergence history tracking for BBOB post-processing.
        Defaults to False.
    [algorithm_specific_params] ([type], optional):
        [Description with BBOB tuning guidance]
        Defaults to [value].
```

### 7. Attributes

List all instance variables, `self.seed` is **required**:

```python
Attributes:
    func (Callable[[ndarray], float]):
        The objective function being optimized.
    lower_bound (float):
        Lower search space boundary.
    upper_bound (float):
        Upper search space boundary.
    dim (int):
        Problem dimensionality.
    max_iter (int):
        Maximum number of iterations.
    seed (int):
        **REQUIRED** Random seed for reproducibility (BBOB compliance).
    population_size (int):
        Number of individuals in population.
    track_history (bool):
        Whether convergence history is tracked.
    history (dict[str, list]):
        Optimization history if track_history=True. Contains:
        - 'best_fitness': list[float] - Best fitness per iteration
        - 'best_solution': list[ndarray] - Best solution per iteration
        - 'population_fitness': list[ndarray] - All fitness values
        - 'population': list[ndarray] - All solutions
    [algorithm_specific_attrs] ([type]):
        [Description]
```

### 8. Methods

Document the search method signature:

```python
Methods:
    search() -> tuple[np.ndarray, float]:
        Execute optimization algorithm.

        Returns:
            tuple[np.ndarray, float]:
                - best_solution (np.ndarray): Best solution found, shape (dim,)
                - best_fitness (float): Fitness value at best_solution

        Raises:
            ValueError:
                If search space is invalid or function evaluation fails.

        Notes:
            - Modifies self.history if track_history=True
            - Uses self.seed for all random number generation
            - BBOB: Returns final best solution after max_iter or convergence
```

### 9. References

Provide DOI links and COCO data archive:

```python
References:
    [1] Author1, A., Author2, B. (YEAR). "Algorithm Name: Description."
        _Journal Name_, Volume(Issue), Pages.
        https://doi.org/10.xxxx/xxxxx

    [2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
        "COCO: A platform for comparing continuous optimizers in a black-box setting."
        _Optimization Methods and Software_, 36(1), 114-144.
        https://doi.org/10.1080/10556788.2020.1808977

    **COCO Data Archive**:
        - Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
        - Algorithm data: [URL to algorithm-specific COCO results if available]
        - Code repository: https://github.com/Anselmoo/useful-optimizer

    **Implementation**:
        - Original paper code: [URL if different from this implementation]
        - This implementation: Based on [1] with modifications for BBOB compliance
```

### 10. See Also

Link related algorithms with BBOB comparisons:

```python
See Also:
    [RelatedAlgorithm1]: Similar algorithm with [key difference]
        BBOB Comparison: [Brief performance notes on sphere/rosenbrock/ackley]

    [RelatedAlgorithm2]: [Relationship description]
        BBOB Comparison: Generally [faster/slower/more robust] on [function classes]

    AbstractOptimizer: Base class for all optimizers
    opt.benchmark.functions: BBOB-compatible test functions

    Related BBOB Algorithm Classes:
        - Evolutionary: GeneticAlgorithm, DifferentialEvolution
        - Swarm: ParticleSwarm, AntColony
        - Gradient: AdamW, SGDMomentum
```

### 11. Notes

Include complexity analysis, BBOB performance, and reproducibility:

```python
Notes:
    **Computational Complexity**:
        - Time per iteration: $O(\text{[expression]})$
        - Space complexity: $O(\text{[expression]})$
        - BBOB budget usage: _[Typical percentage of $\text{dim} \times 10000$ budget needed]_

    **BBOB Performance Characteristics**:
        - **Best function classes**: [Unimodal/Multimodal/Ill-conditioned/...]
        - **Weak function classes**: [Function types where algorithm struggles]
        - Typical success rate at 1e-8 precision: **[X]%** (dim=5)
        - Expected Running Time (ERT): [Comparative notes vs other algorithms]

    **Convergence Properties**:
        - Convergence rate: [Linear/Quadratic/Exponential]
        - Local vs Global: [Tendency for local/global optima]
        - Premature convergence risk: **[High/Medium/Low]**

    **Reproducibility**:
        - **Deterministic**: [Yes/No] - Same seed guarantees same results
        - **BBOB compliance**: seed parameter required for 15 independent runs
        - Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
        - RNG usage: `numpy.random.default_rng(self.seed)` throughout

    **Implementation Details**:
        - Parallelization: [Not supported/Supported via `[method]`]
        - Constraint handling: [Clamping to bounds/Penalty/Repair]
        - Numerical stability: [Considerations for floating-point arithmetic]

    **Known Limitations**:
        - [Any known issues or limitations specific to this implementation]
        - BBOB known issues: [Any BBOB-specific challenges]

    **Version History**:
        - v0.1.0: Initial implementation
        - [vX.X.X]: [Changes relevant to BBOB compliance]
```

## Validation Checklist

Before finalizing optimizer docstring, verify:

- [ ] All 11 sections present in correct order
- [ ] Algorithm metadata table complete with all 9 fields
- [ ] Mathematical formulation includes LaTeX equations
- [ ] Hyperparameters table includes BBOB recommendations
- [ ] BBOB benchmark settings specify dimensions 2,3,5,10,20,40
- [ ] Example includes working doctest with `seed=42`
- [ ] Args section documents `seed` parameter with BBOB guidance
- [ ] Attributes section includes `self.seed` (required)
- [ ] References include DOI and COCO data archive link
- [ ] Notes section includes reproducibility requirements
- [ ] Docstring follows Google style convention (as per pyproject.toml)

## BBOB Metadata Requirements

For full COCO/BBOB compliance, implementations must:

1. **Support seeded randomization**: Accept `seed` parameter and use for all RNG
2. **Track convergence history**: Implement `track_history` flag and populate `self.history`
3. **Return standardized output**: `tuple[np.ndarray, float]` from `search()`
4. **Handle boundary constraints**: Document boundary handling approach
5. **Support standard dimensions**: Test on dims 2, 3, 5, 10, 20, 40
6. **Provide 15 independent runs**: Seeds 0-14 for statistical significance
7. **Document budget usage**: Specify typical function evaluations needed
8. **Include performance notes**: Strengths/weaknesses on BBOB function classes

## Python Code Block Template

```python
class [AlgorithmName](AbstractOptimizer):
    """[Algorithm Full Name] ([ACRONYM]) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | [Full algorithm name]                    |
        | Acronym           | [SHORT]                                  |
        | Year Introduced   | [YYYY]                                   |
        | Authors           | [Last, First; ...]                       |
        | Algorithm Class   | [Category]                               |
        | Complexity        | O([expression])                          |
        | Properties        | [Comma-separated properties]             |
        | Implementation    | Python 3.10+                             |
        | COCO Compatible   | Yes                                      |

    Mathematical Formulation:
        Core update equation:

            $$
            x_{t+1} = x_t + v_t
            $$

        where:
            - $x_t$ is the position at iteration $t$
            - $v_t$ is the velocity/step at iteration $t$

        **Constraint handling**: Clamping to bounds

    Hyperparameters:
        [Table with defaults and BBOB recommendations, use **bold** for emphasis]

    COCO/BBOB Benchmark Settings:
        [Standard benchmark configuration]

    Example:
        [Working doctest examples with seed=42]

    Args:
        [All parameters with BBOB guidance, use `code` for parameter names]

    Attributes:
        [All instance variables including self.seed]

    Methods:
        search() -> tuple[np.ndarray, float]:
            Execute optimization algorithm.

    References:
        [Citations with DOI and COCO links, use _italic_ for journal names]

    See Also:
        [Related algorithms with BBOB comparisons]

    Notes:
        [Use **bold** for section headers, `code` for technical terms,
         _italic_ for emphasis, and LaTeX $...$ for math]
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = DEFAULT_MAX_ITERATIONS,
        seed: int | None = None,
        population_size: int = DEFAULT_POPULATION_SIZE,
        track_history: bool = False,
        # Algorithm-specific parameters
    ) -> None:
        """Initialize the [AlgorithmName] optimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
            track_history=track_history,
        )
        # Initialize algorithm-specific attributes

    def search(self) -> tuple[np.ndarray, float]:
        """Execute [Algorithm] optimization.

        Returns:
            tuple[np.ndarray, float]: Best solution found and its fitness value.
        """
        # Implementation
```

## Reproducibility Requirements

To ensure full BBOB reproducibility compliance:

1. **Seed Management**:
   - Accept `seed` parameter in `__init__`
   - Use `np.random.default_rng(self.seed)` for all random operations
   - Document seed range for BBOB (0-14 for 15 runs)

2. **History Tracking**:
   - Implement `track_history` flag
   - Populate `self.history` dict with keys: `best_fitness`, `best_solution`, `population_fitness`, `population`
   - Update history each iteration when enabled

3. **Deterministic Execution**:
   - Same seed must produce identical results across runs
   - Document any non-deterministic elements (if unavoidable)

4. **Standard Output**:
   - Return `tuple[np.ndarray, float]` from `search()`
   - First element: best solution (shape: (dim,))
   - Second element: best fitness value (scalar float)

5. **Documentation**:
   - Include all 11 required sections
   - Provide working doctest with `seed=42`
   - Document BBOB-specific tuning recommendations

## Usage Guidelines

### For Algorithm Implementers

1. Copy this template when creating new optimizer
2. Replace all `[placeholders]` with algorithm-specific content
3. Run validation checklist before committing
4. Test doctests with: `uv run python -m doctest opt/[category]/[module].py`

### For Documentation Reviewers

1. Verify all 11 sections present
2. Check BBOB metadata completeness
3. Validate doctest examples run successfully
4. Confirm `seed` parameter documented and implemented
5. Review BBOB performance claims against benchmark results

## References

- COCO/BBOB Documentation: https://github.com/numbbo/coco
- COCO Platform: https://coco-platform.org/
- BBOB Test Suite: https://coco-platform.org/testsuites/bbob/overview.html
- Data Archive: https://coco-platform.org/testsuites/bbob/data-archive.html
- Hansen et al. (2021): https://doi.org/10.1080/10556788.2020.1808977
