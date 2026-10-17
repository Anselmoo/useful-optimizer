# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Docs charts: zoom, pan, box zoom, reset and PNG export on the convergence, ECDF
  and violin charts (Ctrl/⌘ + scroll or pinch zooms; a plain scroll still scrolls
  the page). The 3D landscape works again with a normalised height, per-function
  domains, a linear/log height option, click-to-enable scroll zoom, a working
  function selector and a reset button.
- Convergence charts show the median with its interquartile band (or mean ± σ)
  against true iterations or function evaluations.
- `AbstractOptimizer.enable_history_tracking()` to switch on history after
  construction; the benchmark runner uses it so every optimizer reports a history.
- Release tooling checks via repo-release-tools: version targets across the
  package, benchmark module, README and docs nav, plus fingerprints for the
  generated optimizer index and sidebar.

### Fixed

- `search()` of 16 optimizers never finalised `history` because the final-state
  block had ended up inside a docstring: `SGD`, `SGDMomentum`, `ADAMOptimization`,
  `AMSGrad`, `AdaMax`, `Nadam`, `NesterovAcceleratedGradient`, `RMSprop`,
  `AugmentedLagrangian`, `BarrierMethodOptimizer`, `PenaltyMethodOptimizer`,
  `ArtificialFishSwarm`, `DandelionOptimizer`, `DragonflyOptimizer`,
  `HoneyBadgerAlgorithm` and `MothSearchAlgorithm`.
- Optimizers that recorded their history before the first evaluation (e.g.
  `CatSwarmOptimization`, `ParticleFilter`, `ShuffledFrogLeapingAlgorithm`) raised
  a shape error with tracking on; entries without a finite best are now skipped.
- Benchmark results had convergence history for only 4 of 13 optimizers; the
  runner now enables it for all of them. Aggregated histories also store their
  true iteration indices, so the x-axis no longer shows 0-99.
- The ECDF chart no longer plots the budget 0 on a log axis, draws a step curve,
  and counts every (run, target) pair instead of "any run reached the target".
- The violin chart is a real density violin, its checkboxes work, and the points
  no longer move on every redraw. Charts reflow when their container resizes.
- Refreshed benchmark data now triggers the docs deployment.
- Docs CI no longer fails on pull requests after the weekly benchmark refresh:
  the benchmark workflow now refreshes the artifact lock together with the data.

## [0.1.2] - 2024-08-11

### Changed

- Version bump to activate Zenodo archiving.
