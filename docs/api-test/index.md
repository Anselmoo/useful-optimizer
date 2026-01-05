---
title: API Integration Test - AntColony
---

<script setup lang="ts">
import { data as apiData } from '../.vitepress/loaders/api.data'
import APIDoc from '../.vitepress/theme/components/APIDoc.vue'
import { computed } from 'vue'

// Get the AntColony class from swarm_intelligence module
const antColonyClass = computed(() => {
  const swarmModule = apiData.modules.swarm_intelligence
  if (!swarmModule || !swarmModule.classes) return null
  
  return swarmModule.classes.find(c => c.name === 'AntColony')
})

const hasData = computed(() => !!antColonyClass.value)
</script>

# API Integration Test: AntColony

This page demonstrates the integration of Griffe-generated JSON API data with VitePress data loaders and Vue components.

<div v-if="hasData">
  <div class="tip custom-block">
    <p class="custom-block-title">✅ API Data Loaded Successfully</p>
    <p>The data loader successfully parsed the Griffe JSON and found the AntColony class with {{ antColonyClass.parameters?.length || 0 }} parameters and {{ antColonyClass.methods?.length || 0 }} methods.</p>
  </div>

  <APIDoc :classDoc="antColonyClass" />
</div>

<div v-else class="danger custom-block">
  <p class="custom-block-title">❌ API Data Not Available</p>
  <p>Failed to load AntColony class from API data. Check the data loader configuration.</p>
  <details>
    <summary>Debug Information</summary>
    <pre>{{ JSON.stringify(apiData, null, 2) }}</pre>
  </details>
</div>

## Integration Status

- **Data Loader**: `docs/.vitepress/loaders/api.data.ts`
- **Type Definitions**: `docs/.vitepress/types/griffe.d.ts`
- **Component**: `docs/.vitepress/theme/components/APIDoc.vue`
- **Source Data**: `docs/api/full_api.json`

## Next Steps

If the API documentation renders above:
1. ✅ Data loader is correctly parsing Griffe JSON
2. ✅ TypeScript types match the structure
3. ✅ APIDoc component is properly integrated
4. Ready to apply to actual algorithm pages

## See Also

- [Usage Guide](/api-test/USAGE_GUIDE) - How to use this in your algorithm pages
- [Root Level Test](/api-data-test) - Simple data loading test
