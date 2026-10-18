# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking: documentation URLs.** A page's slug is now the optimizer's file stem
  (`whale_optimization_algorithm.py` is `/algorithms/swarm-intelligence/whale-optimization-algorithm`),
  and its title and acronym come from the *Algorithm Metadata* table. 65 of 117 pages moved
  (for example `whale`, `cuckoo`, `tabu`, `genetic`); the old URLs are not redirected.
- **Breaking: module paths.** `opt.probabilistic.parzen_tree_stimator` is now
  `parzen_tree_estimator` and `opt.metaheuristic.variable_neighbourhood_search` is now
  `variable_neighborhood_search`. The classes and the exports from `opt` and the category
  packages are unchanged.
- Acronyms are unique: Bee Algorithm `BEEA`, Artificial Gorilla Troops `AGTO`, Squirrel Search `SQSA`.
- The 11 identical 4 MB Griffe dumps are no longer committed; `docs/api/full_api.json` is
  generated in CI.

### Added

- `RandomSearch` (`opt.classical`, acronym `RS`): pure uniform random search over the search box,
  the reference baseline for the statistical tests in the benchmark suite. It forwards
  `track_history`, `target_precision` and `f_opt` to the base class, uses one seeded generator
  and spends exactly `population_size * iterations` evaluations. The library now has 118 optimizers.
- Generated category index pages and an algorithm catalog (all 118 optimizers are in the
  sidebar, up from 23), generated API pages for all ten categories, a generated README
  catalog and package tree, and `changelog` and `contributing` pages.
- `optimizers.json` carries the full `references` list (with DOIs) of every optimizer;
  `reference` is now filled for 103 of 117 instead of 48.
- Validation is blocking: `unified_validator.py --all` checks every optimizer, requires unique
  acronyms and fails on a new placeholder citation. `scripts/test/test_docs_consistency.py`
  checks that source files, pages, JSON entries and sidebar links agree, that every
  `from opt... import ...` in the docs and README resolves, and that advertised counts are right.
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

- The Quickstart and guides imported names that do not exist (`CMAES`, `AugmentedLagrangianMethod`,
  `opt.abstract_optimizer`) and the constrained example passed a `constraints=` argument the class
  does not take; they now run. The README listed 54 of 117 optimizers with links to the
  pre-`src/` layout.
- 21 optimizers whose `References` entry was a stand-in now carry the citation their module
  docstring already had; 14 still lack a citation and are tracked in
  `scripts/placeholder_references.txt`.
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
