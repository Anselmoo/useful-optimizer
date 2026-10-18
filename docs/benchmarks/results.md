# Benchmark Results

Every optimizer of the library that minimises a single scalar objective is benchmarked with the
[same protocol](./methodology): an identical budget of function evaluations per run, independent
runs with fixed seeds, and a [random search](/algorithms/classical/random-search) baseline that
all statistical tests refer to. The tables and charts on this page read the published benchmark
data in your browser; nothing on this page is typed in by hand.

## Average ranks

Within each function and dimension the optimizers are ranked by their final precision
$|f - f^*|$ in every run; the table shows the average rank over all of these blocks (1 is best,
final errors below the precision floor tie). The Friedman test below the table checks whether the
differences between optimizers are larger than expected by chance.

<ClientOnly>
  <RankTable />
</ClientOnly>

Average ranks summarise performance over very different problems, so they say little about a
specific one. Each [algorithm page](/algorithms/) shows the details for its optimizer.

## Example: one optimizer in detail

The panel below is the one embedded in every algorithm page. Choose the function and dimension,
add optimizers to compare with, and switch between convergence, distribution, runtime ECDF,
scaling with the dimension, statistics against the baseline, dispersion of the evaluated points and
the path through the 2D landscape.

<ClientOnly>
  <BenchmarkCharts algorithm="ParticleSwarm" />
</ClientOnly>

## Reading the charts

- **Convergence** — median precision against function evaluations per dimension; the band is the
  interquartile range of the runs.
- **Distribution** — kernel density of the final precision with boxplot and individual runs.
- **ECDF** — share of (run, target) pairs solved within a budget, the COCO runtime ECDF; further
  left and higher is better.
- **Scaling** — median precision, expected running time or wall-clock cost against the dimension.
- **Statistics** — median and quartiles, success rate, ERT and SP1 at one target, cost per
  evaluation, and the one-sided Mann-Whitney U test against the baseline (Holm-adjusted) with
  Cliff's delta. The table can be sorted and downloaded as CSV.
- **Dispersion** — spread of the recently evaluated points. This is a proxy computed from the
  evaluated points, not the diversity of a population, because most optimizers do not expose one.
- **Landscape (2D)** — best-so-far path of the first run on the 2D function.

Zoom: drag to pan, Ctrl/⌘ + scroll or a pinch to zoom, and use the toolbox for box zoom, reset and
PNG export. A plain scroll over a chart keeps scrolling the page.

## Data

The data is produced by the [benchmark pipeline](https://github.com/Anselmoo/useful-optimizer/actions/workflows/benchmark-pipeline.yml)
and published under `/benchmarks/`. `index.json` describes the protocol and lists the cells, each
cell `cells/<function>-<dimension>.json` holds all optimizers of one function and dimension, and
`ranks.json` holds the average ranks. The format is described by the
[JSON schema](https://github.com/Anselmoo/useful-optimizer/blob/main/docs/schemas/benchmark-data-schema.json).

::: warning Breaking change in 0.2.0
`benchmark-results.json`, the previous single-file format, is no longer published.
:::
