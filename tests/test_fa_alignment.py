"""End-to-end test that FA does what the paper claims: the angle between the
FA signal and the BP signal decreases as training proceeds.

If this ever fails, something structural is wrong with the learning rule or
the angle metric. Both are correctness regressions worth catching.
"""
from __future__ import annotations

import numpy as np

from feedback_alignment.linear import LinearConfig, run_linear_trial


def test_fa_angle_decreases_during_training() -> None:
    cfg = LinearConfig(seed=0, n_examples=2000, omega=0.001, eta=1e-3)
    rows = run_linear_trial(cfg, "fa", track_pinv=False)
    angles = np.array([r["angle_delta_h_fa_bp"] for r in rows])

    early = float(angles[:200].mean())
    late = float(angles[-200:].mean())

    # paper's headline qualitative claim
    assert late < early, f"FA alignment should improve (late {late:.3f} < early {early:.3f})"

    # sanity bounds: angle is in [0, pi/2] by construction
    assert 0.0 <= angles.min() and angles.max() <= np.pi / 2 + 1e-9


def test_fa_learns_better_than_no_learning_baseline() -> None:
    """Sanity: FA actually reduces NSE on the linear task."""
    cfg = LinearConfig(seed=0, n_examples=2000, omega=0.001, eta=1e-3)
    rows = run_linear_trial(cfg, "fa", track_pinv=False)
    init_nse = rows[0]["nse"]
    final_nse = rows[-1]["nse"]
    assert final_nse < 0.1 * init_nse, f"FA should reduce NSE 10x (init {init_nse}, final {final_nse})"
