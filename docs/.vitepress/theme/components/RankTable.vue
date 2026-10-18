<script setup lang="ts">
/**
 * RankTable.vue
 *
 * Average rank of every optimizer over all benchmarked function × dimension
 * blocks (rank 1 = smallest final error, ties share the mean rank), with the
 * Friedman test over the blocks. Loads ranks.json itself; plain HTML.
 */
import { computed, onMounted, ref } from 'vue'
import type { Ranks } from '../types/benchmark'
import { loadRanks } from '../utils/benchmarkData'
import { downloadText, formatNumber, toCsv } from '../utils/benchmarkTransforms'

const data = ref<Ranks | null>(null)
const error = ref<string | null>(null)
const category = ref('all')
const query = ref('')
const ascending = ref(true)

onMounted(async () => {
  try {
    data.value = await loadRanks()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
})

const categories = computed(() => [...new Set(data.value?.entries.map(e => e.category))].sort())

const rows = computed(() => {
  const needle = query.value.trim().toLowerCase()
  const entries = (data.value?.entries ?? [])
    .map((entry, index) => ({ ...entry, overall: index + 1 }))
    .filter(e => category.value === 'all' || e.category === category.value)
    .filter(e => !needle || e.name.toLowerCase().includes(needle))
  entries.sort((a, b) => (ascending.value ? 1 : -1) * (a.mean_rank - b.mean_rank))
  return entries
})

const label = (category: string) => category.replace(/_/g, ' ')

const download = () =>
  downloadText(
    'benchmark-average-ranks.csv',
    toCsv(
      ['rank', 'optimizer', 'category', 'mean_rank'],
      (data.value?.entries ?? []).map((e, i) => [i + 1, e.name, e.category, e.mean_rank])
    )
  )
</script>

<template>
  <div class="ranks">
    <p v-if="error" class="ranks__error">Benchmark ranks are not available: {{ error }}</p>
    <p v-else-if="!data">Loading ranks…</p>
    <template v-else>
      <div class="ranks__controls">
        <label>
          Category
          <select v-model="category">
            <option value="all">All</option>
            <option v-for="c in categories" :key="c" :value="c">{{ label(c) }}</option>
          </select>
        </label>
        <label>
          Search
          <input v-model="query" type="search" placeholder="optimizer name">
        </label>
        <button type="button" @click="ascending = !ascending">
          {{ ascending ? 'Best first' : 'Worst first' }}
        </button>
        <button type="button" @click="download">Download CSV</button>
      </div>
      <div class="ranks__scroll">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Optimizer</th>
              <th>Category</th>
              <th>Average rank</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in rows" :key="row.name">
              <td>{{ row.overall }}</td>
              <td class="name">{{ row.name }}</td>
              <td class="name">{{ label(row.category) }}</td>
              <td>{{ formatNumber(row.mean_rank, 3) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="ranks__note">
        In each of the {{ data.blocks.length }} function × dimension blocks the optimizers are ranked by their
        median final precision over the runs; ties (including all errors below the precision floor) share the
        average rank, and the table shows the mean over the blocks.
        <template v-if="data.friedman">
          Friedman test over the {{ data.friedman.blocks }} blocks with all
          {{ data.friedman.optimizers }} optimizers: χ² = {{ formatNumber(data.friedman.chi2, 4) }},
          p = {{ formatNumber(data.friedman.p, 3) }}.
        </template>
      </p>
    </template>
  </div>
</template>

<style scoped>
.ranks {
  background-color: var(--ctp-mocha-surface0, #313244);
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
}

.ranks__controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  color: var(--ctp-mocha-text, #cdd6f4);
  font-size: 0.875rem;
}

.ranks__controls select,
.ranks__controls input,
.ranks__controls button {
  background-color: var(--ctp-mocha-surface1, #45475a);
  border: 1px solid var(--ctp-mocha-surface2, #585b70);
  border-radius: 4px;
  color: var(--ctp-mocha-text, #cdd6f4);
  padding: 4px 10px;
  font-size: 0.875rem;
}

.ranks__controls button {
  cursor: pointer;
}

.ranks__scroll {
  max-height: 560px;
  overflow: auto;
}

table {
  display: table;
  width: 100%;
  margin: 0;
  border-collapse: collapse;
  font-size: 0.8125rem;
  font-variant-numeric: tabular-nums;
}

th,
td {
  padding: 5px 10px;
  border: none;
  border-bottom: 1px solid var(--ctp-mocha-surface1, #45475a);
  text-align: right;
  background: transparent;
}

th {
  position: sticky;
  top: 0;
  background-color: var(--ctp-mocha-surface0, #313244);
  color: var(--ctp-mocha-subtext1, #bac2de);
}

td.name,
th:nth-child(2),
th:nth-child(3) {
  text-align: left;
}

.ranks__note,
.ranks__error {
  margin: 12px 0 0;
  color: var(--ctp-mocha-overlay1, #7f849c);
  font-size: 0.75rem;
}
</style>
