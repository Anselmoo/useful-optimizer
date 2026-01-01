# Chart Components Test

This page demonstrates the ECharts and TresJS visualization components.

## 2D Charts

### ECDF Chart

<ClientOnly>
<ECDFChart
  :data="[
    {
      algorithm: 'PSO',
      budget: [1, 10, 100, 1000, 10000],
      proportion: [0.1, 0.3, 0.6, 0.8, 0.95]
    },
    {
      algorithm: 'DE',
      budget: [1, 10, 100, 1000, 10000],
      proportion: [0.05, 0.2, 0.5, 0.75, 0.9]
    }
  ]"
  title="ECDF: Algorithm Comparison"
/>
</ClientOnly>

### Convergence Chart

<ClientOnly>
<ConvergenceChart
  :data="[
    {
      algorithm: 'PSO',
      iterations: [0, 10, 20, 30, 40, 50],
      mean: [100, 50, 25, 12.5, 6.25, 3.125],
      std: [10, 8, 6, 4, 2, 1]
    },
    {
      algorithm: 'DE',
      iterations: [0, 10, 20, 30, 40, 50],
      mean: [100, 45, 20, 9, 4, 2],
      std: [12, 9, 7, 5, 3, 1.5]
    }
  ]"
  title="Convergence Comparison"
/>
</ClientOnly>

### Violin Plot

<ClientOnly>
<ViolinPlot
  :data="[
    {
      algorithm: 'PSO',
      values: [1.5, 2.3, 1.8, 2.1, 1.9, 2.0, 1.7, 2.2, 1.6, 2.4]
    },
    {
      algorithm: 'DE',
      values: [2.1, 2.8, 2.3, 2.6, 2.4, 2.5, 2.2, 2.7, 2.0, 2.9]
    }
  ]"
  title="Final Fitness Distribution"
/>
</ClientOnly>

## 3D Visualization

### Fitness Landscape

<ClientOnly>
<FitnessLandscape3D
  functionName="ackley"
  :xRange="[-5, 5]"
  :yRange="[-5, 5]"
  :resolution="100"
  colorScale="viridis"
/>
</ClientOnly>

<script setup>
// Components are auto-registered in theme
</script>
