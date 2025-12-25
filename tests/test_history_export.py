from __future__ import annotations

import json

from benchmarks import save_run_history
from opt.benchmark.functions import sphere
from opt.swarm_intelligence.particle_swarm import ParticleSwarm


def test_particle_swarm_history_and_export(tmp_path):
    optimizer = ParticleSwarm(
        func=sphere,
        lower_bound=-5,
        upper_bound=5,
        dim=5,
        max_iter=5,
        population_size=5,
        seed=42,
        track_history=True,
    )

    solution, fitness = optimizer.search()
    assert isinstance(fitness, float)

    # history should have entries
    assert len(optimizer.history.get("best_fitness", [])) > 0
    assert len(optimizer.history.get("best_solution", [])) > 0

    out = tmp_path / "run.json"
    save_run_history(optimizer, str(out))
    assert out.exists()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "best_fitness" in data
    assert "best_solution" in data
    assert data.get("metadata", {}).get("seed") == 42
    assert data.get("final_result", {}).get("best_fitness") is not None
