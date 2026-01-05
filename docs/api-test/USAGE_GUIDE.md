# Using API Documentation in Algorithm Pages

This guide explains how to integrate auto-generated API documentation from Griffe JSON into VitePress algorithm pages.

## Overview

The Schema-to-VitePress integration allows algorithm pages to display parameter tables, methods, and attributes automatically extracted from the Python source code via Griffe.

## Data Flow

```
Python Source Code (opt/swarm_intelligence/ant_colony.py)
    ↓
Griffe Static Analysis
    ↓
JSON Output (docs/api/full_api.json)
    ↓
VitePress Data Loader (docs/.vitepress/loaders/api.data.ts)
    ↓
Vue Component (APIDoc.vue)
    ↓
Algorithm Page (docs/algorithms/*/algorithm-name.md)
```

## Quick Start

### 1. Add Script Setup Block

Add this to the frontmatter/top of your markdown file:

```vue
<script setup lang="ts">
import { data as apiData } from '../.vitepress/loaders/api.data'
import APIDoc from '../.vitepress/theme/components/APIDoc.vue'
import { computed } from 'vue'

// Replace 'AntColony' with your class name
const classDoc = computed(() => {
  // Replace 'swarm_intelligence' with the appropriate category:
  // swarm_intelligence, evolutionary, gradient_based, classical,
  // metaheuristic, physics_inspired, probabilistic, social_inspired,
  // constrained, multi_objective
  const module = apiData.modules.swarm_intelligence
  if (!module || !module.classes) return null
  
  return module.classes.find(c => c.name === 'AntColony')
})
</script>
```

**Important**: The import paths are relative to the markdown file location:
- From `docs/algorithms/swarm-intelligence/*.md`: use `../../.vitepress/loaders/api.data`
- From `docs/algorithms/*.md`: use `../.vitepress/loaders/api.data`
- From `docs/*.md` (root): use `./.vitepress/loaders/api.data`

### 2. Add APIDoc Component

In your markdown content, add the component where you want the API documentation to appear:

```vue
## API Documentation

<div v-if="classDoc">
  <APIDoc :classDoc="classDoc" />
</div>
<div v-else class="warning custom-block">
  <p>API documentation is loading...</p>
</div>
```

## Category Mappings

Map your algorithm's Python module path to the correct category:

| Python Path | Category |
|------------|----------|
| `opt.swarm_intelligence.*` | `swarm_intelligence` |
| `opt.evolutionary.*` | `evolutionary` |
| `opt.gradient_based.*` | `gradient_based` |
| `opt.classical.*` | `classical` |
| `opt.metaheuristic.*` | `metaheuristic` |
| `opt.physics_inspired.*` | `physics_inspired` |
| `opt.probabilistic.*` | `probabilistic` |
| `opt.social_inspired.*` | `social_inspired` |
| `opt.constrained.*` | `constrained` |
| `opt.multi_objective.*` | `multi_objective` |

## Complete Example

Here's a complete example for the Particle Swarm Optimization page located at `docs/algorithms/swarm-intelligence/particle-swarm.md`:

```markdown
---
title: Particle Swarm Optimization
---

<script setup lang="ts">
import { data as apiData } from '../../.vitepress/loaders/api.data'
import APIDoc from '../../.vitepress/theme/components/APIDoc.vue'
import { computed } from 'vue'

const psoClass = computed(() => {
  const module = apiData.modules.swarm_intelligence
  if (!module || !module.classes) return null
  return module.classes.find(c => c.name === 'ParticleSwarm')
})
</script>

# Particle Swarm Optimization

<span class="badge badge-swarm">Swarm Intelligence</span>

Brief description of PSO...

## Algorithm Overview

Detailed explanation...

## Usage

\`\`\`python
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.benchmark.functions import sphere

optimizer = ParticleSwarm(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500
)

best_solution, best_fitness = optimizer.search()
\`\`\`

## API Documentation

<div v-if="psoClass">
  <APIDoc :classDoc="psoClass" />
</div>
<div v-else class="warning custom-block">
  <p>API documentation is loading...</p>
</div>

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
```

## What Gets Displayed

The `APIDoc` component will automatically render:

1. **Class Header**: Class name and base classes
2. **Description**: Docstring from the Python class
3. **Parameters Table**: All `__init__` parameters with types and defaults
4. **Attributes**: Public class attributes with types
5. **Methods**: Public methods with signatures and descriptions

## Troubleshooting

### API Data Not Loading

If `classDoc` is null:
1. Check that the class name matches exactly (case-sensitive)
2. Verify the category name is correct
3. Ensure `docs/api/full_api.json` exists and contains your class
4. Check browser console for errors

### Build Errors

If you get TypeScript errors:
1. Ensure the import path is correct relative to your markdown file location
   - `../../.vitepress/loaders/api.data` from 2 levels deep
   - `../.vitepress/loaders/api.data` from 1 level deep
   - `./.vitepress/loaders/api.data` from docs root
2. Check that `api.data.ts` exports the correct types
3. Run `npm run docs:build` to see full error details

### Missing Data

If some fields are empty:
1. Check the Python source has proper docstrings
2. Verify type annotations are present in `__init__`
3. Run Griffe manually to regenerate JSON: `npm run docs:api`

## Technical Details

### Data Loader

The data loader (`api.data.ts`) processes Griffe JSON and provides:
- `modules`: Record of all optimizer categories
- `categories`: List of category names
- `classIndex`: Quick lookup by class name

### APIDoc Component Props

```typescript
interface ClassDoc {
  name: string
  docstring: string
  bases: string[]
  parameters: Parameter[]
  methods: Method[]
  attributes: Array<{ name: string; annotation: string; description: string }>
}
```

## Best Practices

1. **Keep Manual Content**: Use APIDoc for technical details, but keep algorithm explanations and usage examples in markdown
2. **Fallback UI**: Always provide a loading/error state for the component
3. **Update JSON**: Regenerate API JSON when optimizer signatures change
4. **Consistent Naming**: Use the exact Python class name in the lookup

## See Also

- [API Test Page](/api-data-test) - Live integration example at docs root
- [VitePress Data Loaders](https://vitepress.dev/guide/data-loading)
