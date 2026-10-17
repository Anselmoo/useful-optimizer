<script setup lang="ts">
/**
 * Common card for the benchmark charts: Catppuccin surface, optional
 * description above and a zoom/pan hint below the plot.
 */
import { ZOOM_HINT } from '../utils/useEChart'

withDefaults(defineProps<{ hint?: string | false }>(), { hint: ZOOM_HINT })
</script>

<template>
  <div class="chart-frame">
    <div v-if="$slots.header" class="chart-frame__header">
      <slot name="header" />
    </div>
    <slot />
    <p v-if="hint" class="chart-frame__hint">{{ hint }}</p>
    <div v-if="$slots.footer" class="chart-frame__footer">
      <slot name="footer" />
    </div>
  </div>
</template>

<style scoped>
.chart-frame {
  background-color: var(--ctp-mocha-surface0, #313244);
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
}

.chart-frame__header {
  color: var(--ctp-mocha-subtext0, #a6adc8);
  font-size: 0.875rem;
  margin-bottom: 12px;
}

.chart-frame__hint {
  margin: 8px 0 0;
  color: var(--ctp-mocha-overlay1, #7f849c);
  font-size: 0.75rem;
  text-align: center;
}

.chart-frame__footer {
  margin-top: 12px;
}
</style>
