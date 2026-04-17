from __future__ import annotations

import numpy as np

from src.sensitivity import run_sobol_sensitivity


def test_sobol_is_deterministic_for_same_inputs() -> None:
    r1 = run_sobol_sensitivity(n_samples=128, asset_type="imoex", regime="all", random_seed=42)
    r2 = run_sobol_sensitivity(n_samples=128, asset_type="imoex", regime="all", random_seed=42)

    s1_1 = r1["sobol_df"]["S1"].to_numpy(dtype=float)
    s1_2 = r2["sobol_df"]["S1"].to_numpy(dtype=float)
    assert np.allclose(s1_1, s1_2, rtol=0.0, atol=1e-12)
    assert r1["top_factor"] == r2["top_factor"]
