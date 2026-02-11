# BenchmarkCharts Component Demo

This page demonstrates the `BenchmarkCharts` wrapper component that simplifies displaying benchmark results.

## Simple Usage

Just provide the algorithm name, function, and dimension:

<ClientOnly>
<BenchmarkCharts 
  algorithm="ParticleSwarm" 
  function="shifted_ackley" 
  :dimension="2" 
/>
</ClientOnly>

---

## Comparing Multiple Algorithms

You can compare multiple algorithms by passing the `compareWith` prop:

<ClientOnly>
<BenchmarkCharts 
  algorithm="ParticleSwarm" 
  functionName="shifted_ackley" 
  :dimension="2"
  :compareWith="['DifferentialEvolution']"
/>
</ClientOnly>

---

## Selective Chart Display

Show only specific charts using the boolean props:

<ClientOnly>
<BenchmarkCharts 
  algorithm="ParticleSwarm" 
  functionName="sphere" 
  :dimension="2"
  :showConvergence="true"
  :showECDF="false"
  :showViolin="false"
/>
</ClientOnly>

---

## API Reference

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `algorithm` | `string` | **required** | Name of the optimization algorithm |
| `functionName` | `string` | **required** | Benchmark function name (e.g., 'shifted_ackley', 'rosenbrock') |
| `dimension` | `number` | `2` | Problem dimension |
| `showConvergence` | `boolean` | `true` | Show convergence chart |
| `showECDF` | `boolean` | `true` | Show ECDF chart |
| `showViolin` | `boolean` | `true` | Show violin plot |
| `targetPrecisions` | `number[]` | `[1e-1, 1e-3, 1e-5, 1e-7]` | Target precisions for ECDF |
| `compareWith` | `string[]` | `[]` | Additional algorithms to compare |

### Features

- ✅ Automatic data loading from `/benchmarks/benchmark-results.json`
- ✅ Error handling with user-friendly messages
- ✅ Loading states with spinner
- ✅ Responsive design
- ✅ Multi-algorithm comparison
- ✅ Selective chart display
- ✅ Consistent styling with Catppuccin theme

### Usage in Algorithm Pages

For algorithm documentation pages, you can easily add benchmarks:

\`\`\`markdown
# Particle Swarm Optimization

... algorithm description ...

## Benchmark Results

<ClientOnly>
<BenchmarkCharts 
  algorithm="ParticleSwarm" 
  functionName="shifted_ackley" 
  :dimension="2" 
/>
</ClientOnly>
\`\`\`

### Comparison Example

To compare PSO against Differential Evolution:

\`\`\`markdown
<ClientOnly>
<BenchmarkCharts 
  algorithm="ParticleSwarm" 
  functionName="rosenbrock" 
  :dimension="5"
  :compareWith="['DifferentialEvolution', 'GeneticAlgorithm']"
/>
</ClientOnly>
\`\`\`

---

## See Also

- [Benchmark Data Demo](/benchmark-data-demo) - Manual data transformation example
- [Chart Components Test](/test-charts) - Individual chart component examples
- [Benchmark Visualization](/benchmarks/visualization) - Benchmark suite overview
