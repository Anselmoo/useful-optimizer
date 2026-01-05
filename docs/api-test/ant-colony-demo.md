---
title: Ant Colony Optimization (API Integration Demo)
---

<script setup lang="ts">
import { data as apiData } from '../../.vitepress/loaders/api.data'
import APIDoc from '../../.vitepress/theme/components/APIDoc.vue'
import { computed } from 'vue'

// Get the AntColony class from swarm_intelligence module
const antColonyClass = computed(() => {
  const swarmModule = apiData.modules.swarm_intelligence
  if (!swarmModule || !swarmModule.classes) return null
  
  return swarmModule.classes.find(c => c.name === 'AntColony')
})
</script>

# Ant Colony Optimization

<span class="badge badge-swarm">Swarm Intelligence</span>

Ant Colony Optimization (ACO) Algorithm - **Enhanced with API Integration**

::: tip API Integration Demo
This page demonstrates the **Schema-to-VitePress Integration** where the parameter table and API documentation are automatically generated from Griffe JSON output. The data flows from:
1. `docs/api/full_api.json` (Griffe output)
2. `docs/.vitepress/loaders/api.data.ts` (VitePress data loader)
3. `APIDoc.vue` component (Vue component rendering)
:::

## Algorithm Overview

This module implements the Ant Colony Optimization (ACO) algorithm. ACO is a
population-based metaheuristic that can be used to find approximate solutions to
difficult optimization problems.

In ACO, a set of software agents called artificial ants search for good solutions to a
given optimization problem. To apply ACO, the optimization problem is transformed into
the problem of finding the best path on a weighted graph. The artificial ants
incrementally build solutions by moving on the graph. The solution construction process
 is stochastic and is biased by a pheromone model, that is, a set of parameters
associated with graph components (either nodes or edges) whose values are modified
at runtime by the ants.

ACO is particularly useful for problems that can be reduced to finding paths on
weighted graphs, like the traveling salesman problem, the vehicle routing problem, and
the quadratic assignment problem.

## Usage

```python
from opt.swarm_intelligence.ant_colony import AntColony
from opt.benchmark.functions import sphere

optimizer = AntColony(
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

## API Documentation

<div v-if="antColonyClass">
  <APIDoc :classDoc="antColonyClass" />
</div>
<div v-else class="warning custom-block">
  <p class="custom-block-title">⚠️ API Data Loading</p>
  <p>API documentation is loading or unavailable.</p>
</div>

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`ant_colony.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/ant_colony.py)
:::
