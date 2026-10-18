<script setup lang="ts">
/**
 * StatsTable.vue
 *
 * Per-optimizer statistics of one function and dimension: final precision,
 * success rate and COCO ERT / SP1 at one target, cost, and the one-sided
 * Mann-Whitney U test against the baseline (Holm-adjusted) with Cliff's delta.
 * Plain HTML, so it works without ECharts; sortable and exportable as CSV.
 */
import { computed, ref } from 'vue'
import {
  downloadText,
  formatNumber,
  significanceMarker,
  toCsv,
  type StatsRow
} from '../utils/benchmarkTransforms'

interface Props {
  rows: StatsRow[]
  baseline: string
  /** Target precision the success rate, ERT and SP1 refer to. */
  target: number
  filename?: string
}

const props = withDefaults(defineProps<Props>(), { filename: 'benchmark-statistics' })

type Key = 'name' | 'median' | 'success' | 'ert' | 'microsecondsPerEval' | 'delta'

const sortKey = ref<Key | null>(null)
const ascending = ref(true)

const sorted = computed(() => {
  const key = sortKey.value
  if (!key) return props.rows
  const direction = ascending.value ? 1 : -1
  return [...props.rows].sort((a, b) => {
    const x = a[key]
    const y = b[key]
    if (x === null && y === null) return 0
    if (x === null) return 1 // missing values last, whatever the direction
    if (y === null) return -1
    return typeof x === 'string' ? direction * x.localeCompare(y as string) : direction * (x - (y as number))
  })
})

const sortBy = (key: Key) => {
  if (sortKey.value === key) ascending.value = !ascending.value
  else {
    sortKey.value = key
    // Higher is better for the success rate and delta; lower for everything else.
    ascending.value = key !== 'success' && key !== 'delta'
  }
}

const marker = (key: Key) => (sortKey.value === key ? (ascending.value ? ' ▲' : ' ▼') : '')
const ariaSort = (key: Key) =>
  sortKey.value === key ? (ascending.value ? 'ascending' : 'descending') : 'none'

const percent = (value: number | null) => (value === null ? '–' : `${(value * 100).toFixed(0)}%`)

const pValue = (row: StatsRow) => {
  if (row.name === props.baseline) return 'baseline'
  if (row.pHolm === null) return '–'
  return `${formatNumber(row.pHolm, 2)}${significanceMarker(row.pHolm)}`
}

const runs = (row: StatsRow) =>
  row.runs === row.runsPlanned ? String(row.runs) : `${row.runs}/${row.runsPlanned}`

const download = () => {
  const header = [
    'optimizer', 'runs', 'runs_planned', 'truncated', 'median', 'q1', 'q3', 'best', 'worst',
    `success@${props.target}`, `ert@${props.target}`, `sp1@${props.target}`,
    'us_per_evaluation', 'p', 'p_holm', 'cliffs_delta'
  ]
  const body = sorted.value.map(r => [
    r.name, r.runs, r.runsPlanned, r.truncated, r.median, r.q1, r.q3, r.best, r.worst,
    r.success, r.ert, r.sp1, r.microsecondsPerEval, r.p, r.pHolm, r.delta
  ])
  downloadText(`${props.filename}.csv`, toCsv(header, body))
}
</script>

<template>
  <div class="stats">
    <div class="stats__scroll">
      <table>
        <thead>
          <tr>
            <th :aria-sort="ariaSort('name')">
              <button type="button" @click="sortBy('name')">Optimizer{{ marker('name') }}</button>
            </th>
            <th>Runs</th>
            <th :aria-sort="ariaSort('median')">
              <button type="button" @click="sortBy('median')">Median error{{ marker('median') }}</button>
            </th>
            <th>IQR</th>
            <th :aria-sort="ariaSort('success')">
              <button type="button" @click="sortBy('success')">Solved@{{ formatNumber(target, 1) }}{{ marker('success') }}</button>
            </th>
            <th :aria-sort="ariaSort('ert')">
              <button type="button" title="Expected running time in evaluations (COCO)" @click="sortBy('ert')">ERT{{ marker('ert') }}</button>
            </th>
            <th title="Average evaluations of successful runs, scaled by the success rate">SP1</th>
            <th :aria-sort="ariaSort('microsecondsPerEval')">
              <button type="button" @click="sortBy('microsecondsPerEval')">µs/eval{{ marker('microsecondsPerEval') }}</button>
            </th>
            <th title="One-sided Mann-Whitney U vs the baseline, Holm-adjusted">p (Holm)</th>
            <th :aria-sort="ariaSort('delta')">
              <button type="button" title="Cliff's delta: +1 = every run beats every baseline run" @click="sortBy('delta')">δ{{ marker('delta') }}</button>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in sorted" :key="row.name" :class="{ baseline: row.name === baseline }">
            <td class="name">{{ row.name }}</td>
            <td>
              {{ runs(row) }}
              <span v-if="row.truncated" class="warn" :title="`${row.truncated} run(s) cut off by the time cap`">⏱</span>
            </td>
            <td>{{ formatNumber(row.median) }}</td>
            <td>{{ formatNumber(row.q1, 2) }} – {{ formatNumber(row.q3, 2) }}</td>
            <td>{{ percent(row.success) }}</td>
            <td>{{ formatNumber(row.ert, 3) }}</td>
            <td>{{ formatNumber(row.sp1, 3) }}</td>
            <td>{{ formatNumber(row.microsecondsPerEval, 3) }}</td>
            <td :class="{ significant: significanceMarker(row.pHolm) }">{{ pValue(row) }}</td>
            <td>{{ row.delta === null ? '–' : row.delta.toFixed(2) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="stats__note">
      ERT and SP1 refer to the target precision {{ formatNumber(target, 1) }}; “–” means the target was
      never reached. p: one-sided Mann-Whitney U test “better than {{ baseline }}”, Holm-adjusted over all
      optimizers of this function and dimension (* &lt; 0.05, ** &lt; 0.01, *** &lt; 0.001). δ: Cliff’s delta.
      ⏱ marks runs cut off by the wall-clock cap.
    </p>
    <button type="button" class="stats__download" @click="download">Download CSV</button>
  </div>
</template>

<style scoped>
.stats {
  background-color: var(--ctp-mocha-surface0, #313244);
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
}

.stats__scroll {
  overflow-x: auto;
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
  padding: 6px 10px;
  border: none;
  border-bottom: 1px solid var(--ctp-mocha-surface1, #45475a);
  text-align: right;
  white-space: nowrap;
  background: transparent;
}

th:first-child,
td.name {
  text-align: left;
}

th {
  color: var(--ctp-mocha-subtext1, #bac2de);
  font-weight: 600;
}

th button {
  all: unset;
  cursor: pointer;
}

th button:hover,
th button:focus-visible {
  color: var(--ctp-mocha-mauve, #cba6f7);
}

tr.baseline td {
  color: var(--ctp-mocha-overlay1, #7f849c);
  font-style: italic;
}

td.significant {
  color: var(--ctp-mocha-green, #a6e3a1);
  font-weight: 600;
}

.warn {
  cursor: help;
}

.stats__note {
  margin: 12px 0 8px;
  color: var(--ctp-mocha-overlay1, #7f849c);
  font-size: 0.75rem;
}

.stats__download {
  background-color: var(--ctp-mocha-surface1, #45475a);
  border: 1px solid var(--ctp-mocha-surface2, #585b70);
  border-radius: 4px;
  color: var(--ctp-mocha-text, #cdd6f4);
  padding: 4px 12px;
  font-size: 0.8125rem;
  cursor: pointer;
}

.stats__download:hover {
  border-color: var(--ctp-mocha-mauve, #cba6f7);
}
</style>
