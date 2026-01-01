# Benchmark Visualization Demo

This page demonstrates the chart components using real mock benchmark data that validates against the schema.

<script setup>
import { computed, ref, onMounted } from 'vue'

// Mock data structure - in production, this would be fetched from the API
const mockData = ref({
  metadata: {
    max_iterations: 100,
    n_runs: 10,
    dimensions: [2, 5, 10],
    timestamp: "2024-12-24T15:45:00Z",
    python_version: "3.12.3",
    numpy_version: "1.26.4"
  },
  benchmarks: {
    shifted_ackley: {
      "2": {
        ParticleSwarm: {
          runs: [
            {
              best_fitness: 0.0012345,
              best_solution: [0.001, -0.002],
              n_evaluations: 2000,
              history: {
                best_fitness: [10.5, 5.2, 2.1, 0.8, 0.3, 0.1, 0.05, 0.02, 0.01, 0.0012345],
                mean_fitness: [15.3, 8.7, 4.5, 2.3, 1.2, 0.6, 0.3, 0.15, 0.08, 0.04]
              }
            },
            {
              best_fitness: 0.0023456,
              best_solution: [-0.001, 0.003],
              n_evaluations: 2000,
              history: {
                best_fitness: [11.2, 6.1, 2.8, 1.1, 0.5, 0.2, 0.08, 0.04, 0.015, 0.0023456],
                mean_fitness: [16.1, 9.2, 5.1, 2.8, 1.5, 0.8, 0.4, 0.2, 0.1, 0.05]
              }
            },
            {
              best_fitness: 0.0018765,
              best_solution: [0.002, 0.001],
              n_evaluations: 2000,
              history: {
                best_fitness: [10.8, 5.8, 2.5, 0.9, 0.4, 0.15, 0.06, 0.03, 0.012, 0.0018765],
                mean_fitness: [15.8, 8.9, 4.8, 2.5, 1.3, 0.7, 0.35, 0.18, 0.09, 0.045]
              }
            }
          ],
          statistics: {
            mean_fitness: 0.0018189,
            std_fitness: 0.0004681,
            min_fitness: 0.0012345,
            max_fitness: 0.0023456,
            median_fitness: 0.0018765,
            q1_fitness: 0.0015555,
            q3_fitness: 0.0021111
          },
          success_rate: 1.0
        },
        DifferentialEvolution: {
          runs: [
            {
              best_fitness: 0.0009876,
              best_solution: [0.0005, -0.0008],
              n_evaluations: 2000,
              history: {
                best_fitness: [12.3, 6.5, 3.2, 1.5, 0.7, 0.3, 0.12, 0.05, 0.02, 0.0009876],
                mean_fitness: [17.2, 10.1, 5.8, 3.1, 1.8, 0.9, 0.45, 0.22, 0.11, 0.055]
              }
            },
            {
              best_fitness: 0.0015432,
              best_solution: [-0.0007, 0.0009],
              n_evaluations: 2000,
              history: {
                best_fitness: [11.8, 6.2, 2.9, 1.3, 0.6, 0.25, 0.1, 0.045, 0.018, 0.0015432],
                mean_fitness: [16.5, 9.5, 5.3, 2.9, 1.6, 0.85, 0.42, 0.21, 0.105, 0.052]
              }
            },
            {
              best_fitness: 0.0011234,
              best_solution: [0.0003, 0.0005],
              n_evaluations: 2000,
              history: {
                best_fitness: [12.0, 6.3, 3.0, 1.4, 0.65, 0.28, 0.11, 0.048, 0.019, 0.0011234],
                mean_fitness: [16.8, 9.8, 5.5, 3.0, 1.7, 0.88, 0.43, 0.215, 0.108, 0.054]
              }
            }
          ],
          statistics: {
            mean_fitness: 0.0012181,
            std_fitness: 0.0002379,
            min_fitness: 0.0009876,
            max_fitness: 0.0015432,
            median_fitness: 0.0011234,
            q1_fitness: 0.0010555,
            q3_fitness: 0.0013333
          },
          success_rate: 1.0
        }
      }
    }
  }
})

// Extract shifted_ackley data for dimension 2
const ackleyData = computed(() => mockData.value.benchmarks.shifted_ackley['2'])
const optimizers = computed(() => Object.keys(ackleyData.value))

// Transform data for ConvergenceChart
const convergenceData = computed(() => {
  return optimizers.value.map(opt => {
    const firstRun = ackleyData.value[opt].runs[0]
    return {
      algorithm: opt,
      iterations: Array.from({length: firstRun.history.best_fitness.length}, (_, i) => i * 10),
      mean: firstRun.history.best_fitness,
      std: firstRun.history.mean_fitness.map((m, i) => Math.abs(m - firstRun.history.best_fitness[i]))
    }
  })
})

// Transform data for ViolinPlot
const violinData = computed(() => {
  return optimizers.value.map(opt => ({
    algorithm: opt,
    values: ackleyData.value[opt].runs.map(run => run.best_fitness)
  }))
})

// Transform data for ECDF
const ecdfData = computed(() => {
  return optimizers.value.map(opt => {
    const runs = ackleyData.value[opt].runs
    const fitnessValues = runs.map(r => r.best_fitness).sort((a, b) => a - b)
    const budgets = [100, 500, 1000, 1500, 2000]
    const proportions = budgets.map(budget => {
      // Simulate proportion based on fitness threshold
      const threshold = 0.01 / (budget / 2000)
      return fitnessValues.filter(f => f <= threshold).length / fitnessValues.length
    })

    return {
      algorithm: opt,
      budget: budgets,
      proportion: proportions
    }
  })
})

// Metadata display
const metadata = computed(() => mockData.value.metadata)
</script>

## Dataset Information

**Benchmark Suite Metadata:**
- **Max Iterations:** {{ metadata.max_iterations }}
- **Number of Runs:** {{ metadata.n_runs }}
- **Dimensions:** {{ metadata.dimensions.join(', ') }}
- **Python Version:** {{ metadata.python_version }}
- **NumPy Version:** {{ metadata.numpy_version }}
- **Timestamp:** {{ metadata.timestamp }}

---

## Convergence Analysis

Comparison of **{{ optimizers.join(' vs ') }}** on the **Shifted Ackley** function (dimension 2):

<ClientOnly>
<ConvergenceChart
  :data="convergenceData"
  title="Convergence: Shifted Ackley (2D)"
  xAxisLabel="Iteration"
  yAxisLabel="Best Fitness"
  :logScale="true"
  :showConfidenceBand="true"
/>
</ClientOnly>

---

## Final Fitness Distribution

Statistical distribution of final fitness values across multiple runs:

<ClientOnly>
<ViolinPlot
  :data="violinData"
  title="Final Fitness Distribution: Shifted Ackley (2D)"
  yAxisLabel="Best Fitness"
  :logScale="true"
  :showBoxplot="true"
  :showPoints="true"
/>
</ClientOnly>

---

## ECDF Analysis

Empirical Cumulative Distribution Function showing the proportion of targets reached:

<ClientOnly>
<ECDFChart
  :data="ecdfData"
  title="ECDF: Shifted Ackley (2D)"
  xAxisLabel="Budget (function evaluations)"
  yAxisLabel="Proportion of targets reached"
  :logXAxis="false"
/>
</ClientOnly>

---

## 3D Fitness Landscape

Interactive 3D visualization of the Ackley function:

<ClientOnly>
<FitnessLandscape3D
  functionName="ackley"
  :xRange="[-5, 5]"
  :yRange="[-5, 5]"
  :resolution="80"
  colorScale="viridis"
  :height="500"
/>
</ClientOnly>

---

## Summary Statistics

<div class="stats-grid">
  <div v-for="opt in optimizers" :key="opt" class="stats-card">
    <h3>{{ opt }}</h3>
    <div class="stats-content">
      <div class="stat-item">
        <span class="label">Mean Fitness:</span>
        <span class="value">{{ ackleyData[opt].statistics.mean_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Std Deviation:</span>
        <span class="value">{{ ackleyData[opt].statistics.std_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Best Fitness:</span>
        <span class="value success">{{ ackleyData[opt].statistics.min_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Worst Fitness:</span>
        <span class="value">{{ ackleyData[opt].statistics.max_fitness.toExponential(4) }}</span>
      </div>
      <div class="stat-item">
        <span class="label">Success Rate:</span>
        <span class="value">{{ (ackleyData[opt].success_rate * 100).toFixed(0) }}%</span>
      </div>
      <div class="stat-item">
        <span class="label">Number of Runs:</span>
        <span class="value">{{ ackleyData[opt].runs.length }}</span>
      </div>
    </div>
  </div>
</div>

<style scoped>
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.stats-card {
  background-color: var(--ctp-mocha-surface0, #313244);
  border-radius: 8px;
  padding: 20px;
  border: 1px solid var(--ctp-mocha-surface1, #45475a);
}

.stats-card h3 {
  margin: 0 0 16px 0;
  color: var(--ctp-mocha-mauve, #cba6f7);
  font-size: 1.2rem;
  border-bottom: 2px solid var(--ctp-mocha-surface1, #45475a);
  padding-bottom: 8px;
}

.stats-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
}

.stat-item .label {
  color: var(--ctp-mocha-subtext0, #a6adc8);
  font-size: 0.9rem;
}

.stat-item .value {
  color: var(--ctp-mocha-text, #cdd6f4);
  font-weight: 600;
  font-family: monospace;
}

.stat-item .value.success {
  color: var(--ctp-mocha-green, #a6e3a1);
}
</style>
