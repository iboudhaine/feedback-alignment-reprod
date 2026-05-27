"""Task 1: linear function approximation (paper's Fig 1 and Fig 4).

Architecture: 30 -> 20 -> 10, all linear, no biases (paper formulation y = W * W0 * x).
Target T ~ Uniform[-1, 1]; inputs x ~ N(0, I); feedback B ~ Uniform[-0.5, 0.5].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .algorithms import Algorithm, step
from .core import Activation, angle, split_rngs, uniform
from .io import build_slug, stream_rows, write_rows
from .layers import init_mlp


@dataclass
class LinearConfig:
    ni: int = 30
    nh: int = 20
    no: int = 10
    n_examples: int = 2000
    eta: float = 1e-3
    omega: float = 0.01          # init scale for W0, W (paper Task 1 spec)
    b_scale: float = 0.5         # feedback matrix scale (paper Task 1 spec)
    seed: int = 0

    def slug(self) -> str:
        return build_slug([
            ("seed", self.seed),
            ("eta", self.eta),
            ("omega", self.omega),
            ("b", self.b_scale),
        ])


def _make_dataset(cfg: LinearConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = rng.uniform(-1.0, 1.0, size=(cfg.no, cfg.ni)).astype(np.float64)
    X = rng.normal(0.0, 1.0, size=(cfg.n_examples, cfg.ni)).astype(np.float64)
    Y = X @ T.T
    return T, X, Y


def run_linear_trial(
    cfg: LinearConfig,
    alg: Algorithm,
    *,
    track_pinv: bool = False,
) -> list[dict]:
    """Run one trial of the linear task and return per-step metrics.

    Always computes both the BP and FA pre-activation signals at the hidden
    layer, so the FA-vs-BP angle is logged on every run regardless of `alg`.
    If `track_pinv` is set (Fig 4), also logs the angle to the pseudo-backprop
    signal pinv(W) e.
    """
    rngs = split_rngs(cfg.seed, ["data", "init", "feedback"])
    _, X, Y = _make_dataset(cfg, rngs["data"])

    mlp = init_mlp(
        sizes=[cfg.ni, cfg.nh, cfg.no],
        activations=[Activation.LINEAR, Activation.LINEAR],
        init_scale=cfg.omega,
        rng=rngs["init"],
        use_bias=False,
    )
    feedback = [uniform((cfg.nh, cfg.no), cfg.b_scale, rngs["feedback"])]

    rows: list[dict] = []
    for t in range(cfg.n_examples):
        # Capture pinv(W_top) before step() mutates W.
        pinv_W_top = np.linalg.pinv(mlp.layers[-1].W) if track_pinv else None

        result = step(mlp, X[t], Y[t], alg=alg, eta=cfg.eta, feedback_matrices=feedback)

        assert result.signals_fa is not None  # feedback is always supplied here
        row: dict[str, float | int] = {
            "step": t,
            "nse": result.nse,
            "angle_delta_h_fa_bp": angle(result.signals_fa[0], result.signals_bp[0]),
        }
        if pinv_W_top is not None:
            # Linear output, so the top-layer signal equals e.
            # The pseudo-backprop hidden signal is pinv(W) @ e.
            e = Y[t] - result.y
            dh_pbp = pinv_W_top @ e
            row["angle_delta_h_fa_pbp"] = angle(result.signals_fa[0], dh_pbp)
        rows.append(row)
    return rows


def run_fig1(cfg: LinearConfig, alg: Algorithm, out_csv: Path) -> list[dict]:
    """Reproduce Fig 1: BP vs FA learning curves + alignment angle."""
    rows = run_linear_trial(cfg, alg, track_pinv=False)
    write_rows(out_csv, rows)
    return rows


def run_fig4(
    omega_list: list[float],
    n_trials: int,
    eta: float,
    seed: int,
    out_csv: Path,
    n_examples: int = 2000,
) -> None:
    """Reproduce Fig 4: FA-only sweep over the init scale omega.

    Fig 4 is FA-only by construction, so no alg argument is exposed.
    Writes one row per (omega, trial, step).
    """
    fieldnames = ["omega", "trial", "step", "nse", "angle_delta_h_fa_bp", "angle_delta_h_fa_pbp"]
    f, writer = stream_rows(out_csv, fieldnames)
    try:
        for omega in omega_list:
            for trial in range(n_trials):
                cfg_t = LinearConfig(
                    eta=eta,
                    omega=omega,
                    seed=seed + 1000 * trial,
                    n_examples=n_examples,
                )
                rows = run_linear_trial(cfg_t, "fa", track_pinv=True)
                for row in rows:
                    row["omega"] = omega
                    row["trial"] = trial
                    writer.writerow(row)
    finally:
        f.close()
