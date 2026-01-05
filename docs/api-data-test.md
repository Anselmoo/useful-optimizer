---
title: API Data Test
---

<script setup lang="ts">
import { data } from './.vitepress/loaders/api.data'
</script>

# API Data Test

This is a test page at the docs root level to verify data loader imports.

## Data Check

<div v-if="data">
  <p>✅ API Data loaded successfully!</p>
  <p>Categories: {{ data.categories.join(', ') }}</p>
  <p>Number of swarm intelligence classes: {{ data.modules.swarm_intelligence?.classes?.length || 0 }}</p>
</div>
<div v-else>
  <p>❌ API Data failed to load</p>
</div>
