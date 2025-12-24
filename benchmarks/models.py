"""Pydantic models for benchmark data schemas.

Auto-generated from benchmark-data-schema.json (2025-12-24T15:07:25+00:00).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field
from pydantic import RootModel


if TYPE_CHECKING:
    from pydantic import AwareDatetime


class Dimension(RootModel[int]):
    """Dimension constraint model for benchmark data."""
    root: int = Field(..., ge=2)


class Metadata(BaseModel):
    """Metadata for benchmark execution."""
    max_iterations: int = Field(..., ge=1)
    n_runs: int = Field(..., ge=1)
    dimensions: list[Dimension]
    timestamp: AwareDatetime
    python_version: str | None = None
    numpy_version: str | None = None


class History(BaseModel):
    """History tracking data for optimization runs."""
    best_fitness: list[float] | None = None
    mean_fitness: list[float] | None = None


class Run(BaseModel):
    """Individual optimization run results."""
    best_fitness: float
    best_solution: list[float]
    n_evaluations: int
    history: History | None = None


class Statistics(BaseModel):
    """Statistical summary of benchmark results."""
    mean_fitness: float
    std_fitness: float
    min_fitness: float
    max_fitness: float
    median_fitness: float
    q1_fitness: float | None = None
    q3_fitness: float | None = None


class Benchmarks(BaseModel):
    """Benchmark results container."""
    runs: list[Run]
    statistics: Statistics
    success_rate: float = Field(..., ge=0.0, le=1.0)


class BenchmarkDataSchema(BaseModel):
    """Complete benchmark data schema."""
    metadata: Metadata
    benchmarks: dict[str, dict[str, dict[str, Benchmarks]]]
