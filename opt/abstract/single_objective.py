"""Abstract base class for single-objective optimizers.

**COCO/BBOB Compliance Requirements:**
All concrete optimizer implementations inheriting from this class must provide:
- Algorithm metadata (name, version, authors, year, class)
- BBOB benchmark settings (search space, dimensions, runs, seeds)
- Hyperparameter documentation with BBOB-recommended values
- Reproducibility requirements (seed logging, parameter tracking)
- Performance characteristics on BBOB function classes
- Complexity analysis (time/space, function evaluations)

See `.github/prompts/optimizer-docs-template.prompt.md` for complete template.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from opt.abstract.history import HistoryConfig
from opt.abstract.history import OptimizationHistory
from opt.constants import DEFAULT_MAX_ITERATIONS
from opt.constants import DEFAULT_POPULATION_SIZE
from opt.constants import DEFAULT_SEED
from opt.constants import POWER_THIRTY_TWO


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class AbstractOptimizer(ABC):
    """An abstract base class for optimizers with COCO/BBOB compliance support.

    This base class provides the foundation for single-objective optimization algorithms
    with built-in support for COCO/BBOB benchmark requirements including reproducibility,
    history tracking, and standardized interfaces.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
            Must accept a NumPy array and return a scalar.
            BBOB functions available in `opt.benchmark.functions`.
        lower_bound (float): The lower bound of the search space.
            BBOB typical: -5 (most functions), -100 (Rastrigin, Weierstrass).
        upper_bound (float): The upper bound of the search space.
            BBOB typical: 5 (most functions), 100 (Rastrigin, Weierstrass).
        dim (int): The dimensionality of the search space.
            BBOB standard dimensions: 2, 3, 5, 10, 20, 40.
        max_iter (int, optional): The maximum number of iterations.
            BBOB recommendation: 10000 for complete evaluation.
            Defaults to 1000.
        seed (int | None, optional): **REQUIRED for BBOB compliance.** Random seed.
            BBOB requires seeds 0-14 for 15 independent runs.
            If None, generates a random seed. Defaults to None.
        population_size (int, optional): The number of individuals in the population.
            BBOB recommendation: 10*dim for population-based algorithms.
            Defaults to 100.
        track_history (bool, optional): Whether to track optimization history.
            When enabled, stores convergence data for visualization and COCO postprocessing.
            Defaults to False.

    Attributes:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int): The maximum number of iterations for the optimization process.
        seed (int): **REQUIRED for BBOB compliance.** The random seed.
            Used for all random operations to ensure reproducibility.
        population_size (int): The number of individuals in the population.
        track_history (bool): Whether to track optimization history.
        history (dict[str, list]): Dictionary containing optimization history if track_history is True.
            Contains keys: 'best_fitness', 'best_solution', 'population_fitness', 'population'.

    Methods:
        search() -> tuple[ndarray, float]: Perform the optimization search.

    Returns:
        tuple[ndarray, float]: Tuple containing the best solution found (shape: (dim,))
        and its corresponding fitness value (scalar).

    Notes:
        **BBOB Standard Settings:**
        - Search space bounds: Typically [-5, 5] for most functions
        - Evaluation budget: dim * 10000 function evaluations
        - Independent runs: 15 (using seeds 0-14)
        - Target precision: 1e-8

        **Reproducibility:**
            - Same seed must produce identical results across runs
            - All random operations use `np.random.default_rng(self.seed)`
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = DEFAULT_MAX_ITERATIONS,
        seed: int | None = None,
        population_size: int = DEFAULT_POPULATION_SIZE,
        track_history: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the optimizer."""
        self.func = func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.max_iter = max_iter
        if seed is None:
            self.seed = np.random.default_rng(DEFAULT_SEED).integers(
                0, 2**POWER_THIRTY_TWO
            )
        else:
            self.seed = seed
        self.population_size = population_size
        self.track_history = track_history
        self._history_buffer = (
            OptimizationHistory(
                max_iter=self.max_iter + 1,  # include final state
                dim=self.dim,
                population_size=self.population_size,
                config=HistoryConfig(
                    track_population=True,
                    track_population_fitness=True,
                    max_history_size=self.max_iter + 1,
                ),
            )
            if track_history
            else None
        )
        self.history: dict[str, list] = (
            {
                "best_fitness": [],
                "best_solution": [],
                "population_fitness": [],
                "population": [],
            }
            if track_history
            else {}
        )

    def _record_history(
        self,
        best_fitness: float,
        best_solution: ndarray,
        population_fitness: ndarray | None = None,
        population: ndarray | None = None,
    ) -> None:
        """Record iteration history using preallocated storage."""
        if not self.track_history or self._history_buffer is None:
            return
        self._history_buffer.record(
            best_fitness=best_fitness,
            best_solution=best_solution,
            population_fitness=population_fitness,
            population=population,
        )

    def _finalize_history(self) -> None:
        """Convert preallocated history to list-based format for consumers."""
        if not self.track_history or self._history_buffer is None:
            return
        self.history = self._history_buffer.to_dict()

    def export_benchmark_json(
        self,
        result: dict,
        path: str | None = None,
        schema_path: str | None = "docs/schemas/benchmark-data-schema.json",
    ) -> str:
        """Export benchmark result dict to JSON and optionally validate against a schema.

        Args:
            result (dict): The benchmark result dictionary to export.
            path (str | None): Path to write JSON file. If None, a default path is chosen.
            schema_path (str | None): Optional path to a JSON Schema for validation.

        Returns:
            str: Path to written JSON file.
        """
        import json

        from pathlib import Path

        try:
            out_dir = Path(path).parent if path else Path("benchmarks/output")
        except TypeError:
            out_dir = Path("benchmarks/output")
        out_dir.mkdir(parents=True, exist_ok=True)

        if path is None:
            import time

            timestamp = int(time.time())
            filename = f"{self.__class__.__name__}-{timestamp}-s{self.seed}.json"
            file_path = out_dir / filename
        else:
            file_path = Path(path)

        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

        # If schema validation is requested, try to validate using jsonschema if available
        if schema_path:
            import warnings

            from importlib import util as importlib_util
            from pathlib import Path

            if importlib_util.find_spec("jsonschema") is None:
                warnings.warn(
                    "jsonschema not installed; skipping benchmark JSON validation",
                    stacklevel=2,
                )
            else:
                from jsonschema import Draft7Validator

                with Path(schema_path).open(encoding="utf-8") as s:
                    schema = json.load(s)
                Draft7Validator(schema).validate(result)

        return str(file_path)

    def benchmark(
        self,
        *,
        store: bool = False,
        out_path: str | None = None,
        schema_path: str | None = "docs/schemas/benchmark-data-schema.json",
        quick: bool = True,
        quick_max_iter: int = 10,
        quick_population_size: int | None = None,
        **kwargs: object,
    ) -> dict:
        """Run a short benchmark of this optimizer and optionally store results.

        The method temporarily enables history tracking to collect full iteration data
        and restores the optimizer's previous configuration after execution.

        Args:
            store (bool): If True, write the result JSON to disk and return path metadata.
            out_path (str | None): Explicit output path for JSON export.
            schema_path (str | None): Path to JSON schema used to validate the output.
            quick (bool): When True, run a short internal benchmark (smaller max_iter).
            quick_max_iter (int): Maximum iterations to use in quick mode.
            quick_population_size (int | None): Override population size in quick mode.
            **kwargs: Extra keyword arguments forwarded to `search` where applicable.

        Returns:
            dict: Result dictionary containing metadata, params, best_solution, best_fitness and history (if available).
        """
        import time

        # Save state
        orig_track = self.track_history
        orig_history = self.history.copy() if isinstance(self.history, dict) else {}
        orig_history_buffer = self._history_buffer
        orig_max_iter = self.max_iter
        orig_population = self.population_size

        try:
            # Adjust for quick run
            if quick:
                self.max_iter = min(self.max_iter, quick_max_iter)
                if quick_population_size is not None:
                    self.population_size = quick_population_size

            # Ensure history is enabled for benchmark
            if not self.track_history or self._history_buffer is None:
                self.track_history = True
                self._history_buffer = OptimizationHistory(
                    max_iter=self.max_iter + 1,
                    dim=self.dim,
                    population_size=self.population_size,
                    config=HistoryConfig(
                        track_population=True,
                        track_population_fitness=True,
                        max_history_size=self.max_iter + 1,
                    ),
                )
                self.history = {
                    "best_fitness": [],
                    "best_solution": [],
                    "population_fitness": [],
                    "population": [],
                }

            # Run the optimizer's search implementation
            solution, fitness = self.search(**kwargs)

            # Finalize history
            self._finalize_history()

            result: dict = {
                "algorithm": self.__class__.__name__,
                "seed": int(self.seed),
                "timestamp": int(time.time()),
                "params": {
                    "dim": int(self.dim),
                    "max_iter": int(self.max_iter),
                    "population_size": int(self.population_size),
                },
                "best_solution": (
                    solution.tolist() if hasattr(solution, "tolist") else solution
                ),
                "best_fitness": float(fitness),
                "history": self.history,
            }

            if store:
                # Build schema-compliant artifact
                import datetime
                import sys

                import numpy as np

                from opt.benchmark.utils import export_benchmark_json as export_helper

                metadata = {
                    "max_iterations": int(self.max_iter),
                    "n_runs": 1,
                    "dimensions": [int(self.dim)],
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "python_version": sys.version.split()[0],
                    "numpy_version": np.__version__,
                }

                func_name = getattr(self.func, "__name__", "function")
                artifact = {
                    "metadata": metadata,
                    "benchmarks": {
                        func_name: {
                            str(self.dim): {
                                self.__class__.__name__: {
                                    "runs": [
                                        {
                                            "best_fitness": result.get("best_fitness"),
                                            "best_solution": result.get(
                                                "best_solution"
                                            ),
                                            "n_evaluations": int(self.max_iter),
                                            "history": result.get("history", {}),
                                        }
                                    ],
                                    "statistics": {
                                        "mean_fitness": float(
                                            result.get("best_fitness")
                                        ),
                                        "std_fitness": 0.0,
                                        "min_fitness": float(
                                            result.get("best_fitness")
                                        ),
                                        "max_fitness": float(
                                            result.get("best_fitness")
                                        ),
                                        "median_fitness": float(
                                            result.get("best_fitness")
                                        ),
                                    },
                                    "success_rate": 1.0,
                                }
                            }
                        }
                    },
                }

                path = export_helper(artifact, out_path, schema_path=schema_path)
                return {"path": path, "metadata": metadata}
            return result
        finally:
            # Restore state
            self.max_iter = orig_max_iter
            self.population_size = orig_population
            self.track_history = orig_track
            self._history_buffer = orig_history_buffer
            if orig_track:
                self.history = orig_history

    @abstractmethod
    def search(self) -> tuple[ndarray, float]:
        """Perform the optimization search.

        Returns:
        Tuple containing the best solution found and its corresponding fitness value.
        """
