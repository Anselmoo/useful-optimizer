# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Release tooling checks via repo-release-tools: version targets across the
  package, benchmark module, README and docs nav, plus fingerprints for the
  generated optimizer index and sidebar.

### Fixed

- Docs CI no longer fails on pull requests after the weekly benchmark refresh:
  the benchmark workflow now refreshes the artifact lock together with the data.

## [0.1.2] - 2024-08-11

### Changed

- Version bump to activate Zenodo archiving.
