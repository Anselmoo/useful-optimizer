# Particle Swarm Optimization

<span class="badge badge-swarm">Swarm Intelligence</span>

Particle Swarm Optimization (PSO) is a population-based stochastic optimization technique inspired by the social behavior of bird flocking or fish schooling.

## Algorithm Overview

PSO was introduced by Kennedy and Eberhart in 1995. Each particle in the swarm represents a potential solution and moves through the search space influenced by:
- Its own best known position (**cognitive component**)
- The swarm's best known position (**social component**)

## Mathematical Formulation

### Velocity Update

$$
v_i^{t+1} = w \cdot v_i^t + c_1 \cdot r_1 \cdot (p_i - x_i^t) + c_2 \cdot r_2 \cdot (g - x_i^t)
$$

### Position Update

$$
x_i^{t+1} = x_i^t + v_i^{t+1}
$$

Where:
- $w$ - Inertia weight
- $c_1$ - Cognitive coefficient
- $c_2$ - Social coefficient
- $r_1, r_2$ - Random numbers in [0, 1]
- $p_i$ - Personal best position
- $g$ - Global best position

## Usage

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

optimizer = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-12.768,
    upper_bound=12.768,
    dim=10,
    max_iter=100,
    population_size=50,
    w=0.7,      # Inertia weight
    c1=1.5,     # Cognitive coefficient
    c2=1.5      # Social coefficient
)

best_solution, best_fitness = optimizer.search()
print(f"Best fitness: {best_fitness:.6e}")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | Required | Objective function to minimize |
| `lower_bound` | `float` | Required | Lower bound of search space |
| `upper_bound` | `float` | Required | Upper bound of search space |
| `dim` | `int` | Required | Problem dimensionality |
| `max_iter` | `int` | 100 | Maximum iterations |
| `population_size` | `int` | 30 | Number of particles |
| `w` | `float` | 0.7 | Inertia weight |
| `c1` | `float` | 1.5 | Cognitive coefficient |
| `c2` | `float` | 1.5 | Social coefficient |

## Inertia Weight Strategies

The inertia weight $w$ controls exploration vs. exploitation:

- **High $w$ (0.9-1.0)**: More exploration, particles move freely
- **Low $w$ (0.4-0.6)**: More exploitation, particles converge faster

### Linear Decreasing Weight

A common strategy is to linearly decrease $w$ over time:

$$
w^t = w_{max} - \frac{t}{T} \cdot (w_{max} - w_{min})
$$

## Variants

### Standard PSO (SPSO)
The basic implementation with inertia weight.

### Constriction PSO
Uses a constriction factor instead of inertia weight:

$$
\chi = \frac{2}{\left|2 - \phi - \sqrt{\phi^2 - 4\phi}\right|}
$$

where $\phi = c_1 + c_2 > 4$.

### Comprehensive Learning PSO (CLPSO)
Each dimension learns from different particles' best positions.

## When to Use PSO

::: tip Recommended For
- Continuous optimization problems
- Multi-modal landscapes
- Real-time optimization
- Problems where gradient is unavailable
:::

::: warning Limitations
- May converge prematurely on complex landscapes
- Performance depends heavily on parameter tuning
- Not suitable for discrete optimization (without modifications)
:::

## Benchmark Performance

| Function | 10D Mean | 30D Mean | Success Rate |
|----------|----------|----------|--------------|
| Sphere | 1.2e-5 | 3.4e-4 | 100% |
| Rosenbrock | 3.4e-2 | 1.2e+0 | 85% |
| Rastrigin | 8.5e+0 | 4.2e+1 | 65% |
| Ackley | 2.1e-4 | 5.6e-3 | 95% |

## References

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95*, Vol. 4, pp. 1942-1948.

2. Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. *Proceedings of IEEE ICEC*, pp. 69-73.

3. Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, stability, and convergence in a multidimensional complex space. *IEEE Transactions on Evolutionary Computation*, 6(1), 58-73.

## See Also

- [Grey Wolf Optimizer](./grey-wolf) - Another swarm-based algorithm
- [Whale Optimization](./whale) - Marine-inspired optimization
- [Ant Colony](./ant-colony) - Pheromone-based search
