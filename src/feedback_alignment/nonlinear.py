"""Task 3: nonlinear function approximation.

A fixed teacher network (30-20-10-10, tanh hidden, linear output) generates
targets for inputs drawn from N(0, I). Student networks of depth 3 (30-20-10)
or depth 4 (30-20-10-10) try to match the teacher under BP, FA, or shallow
learning. The teacher regime and the FA feedback scales are picked by manual
search; `nonlinear_sweep` runs the grid that exposes that choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .algorithms import Algorithm, step
from .core import Activation, split_rngs, uniform
from .io import build_slug, stream_rows, write_rows
from .layers import MLP, init_mlp


@dataclass
class NonlinearConfig:
    ni: int = 30
    nh0: int = 20
    nh1: int = 10
    no: int = 10
    steps: int = 1_500_000
    eta: float = 1e-3
    seed: int = 0
    test_size: int = 5000
    eval_every: int = 50_000

    # Teacher regime is selected manually. Accept a scalar (same scale every
    # layer) or a list of three floats for per-layer control.
    target_scale: float | Sequence[float] = 1.0
    # Paper teacher has no biases by default. 0.0 means zero biases.
    # A positive value draws biases uniform in [-teacher_bias_scale, teacher_bias_scale].
    teacher_bias_scale: float = 0.0

    # FA feedback matrix scales. Paper notation: b1 maps into layer 0,
    # b2 maps into layer 1. b2_scale is unused for depth-3 students.
    b1_scale: float | None = None
    b2_scale: float | None = None

    init_scale: float = 0.01  # student weight init

    def slug(self) -> str:
        if isinstance(self.target_scale, (int, float)):
            ts: str | float = float(self.target_scale)
        else:
            ts = "x".join(str(float(x)) for x in self.target_scale)
        return build_slug(
            [
                ("seed", self.seed),
                ("ts", ts),
                ("b1", self.b1_scale),
                ("b2", self.b2_scale),
            ]
        )


# --- architectures


def _student_arch(depth: int, cfg: NonlinearConfig) -> tuple[list[int], list[Activation]]:
    if depth == 3:
        return [cfg.ni, cfg.nh0, cfg.no], [Activation.TANH, Activation.LINEAR]
    if depth == 4:
        return [cfg.ni, cfg.nh0, cfg.nh1, cfg.no], [
            Activation.TANH,
            Activation.TANH,
            Activation.LINEAR,
        ]
    raise ValueError(f"model depth must be 3 or 4, got {depth}")


def _teacher_arch(cfg: NonlinearConfig) -> tuple[list[int], list[Activation]]:
    return [cfg.ni, cfg.nh0, cfg.nh1, cfg.no], [Activation.TANH, Activation.TANH, Activation.LINEAR]


def _target_scales(cfg: NonlinearConfig) -> list[float]:
    """Normalize target_scale to a per-layer list (length = teacher depth = 3)."""
    if isinstance(cfg.target_scale, (int, float)):
        return [float(cfg.target_scale)] * 3
    scales = [float(x) for x in cfg.target_scale]
    if len(scales) != 3:
        raise ValueError(f"target_scale list must have length 3, got {len(scales)}")
    return scales


# --- builders


def _build_teacher(cfg: NonlinearConfig, rng: np.random.Generator) -> MLP:
    sizes, acts = _teacher_arch(cfg)
    scales = _target_scales(cfg)
    use_bias = cfg.teacher_bias_scale > 0
    return init_mlp(
        sizes=sizes,
        activations=acts,
        init_scale=scales,
        rng=rng,
        use_bias=use_bias,
        bias_scale=cfg.teacher_bias_scale if use_bias else None,
    )


def _build_feedback(
    cfg: NonlinearConfig, model_depth: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """FA feedback matrices for the student.

    Returned list aligns with `algorithms.step`'s expectation: entry i has shape
    `(layers[i].n_out, layers[i+1].n_out)` and replaces `layers[i+1].W.T` in the
    FA error path. Naming maps to the paper as:
        depth-3 student: [B1]      where B1: (nh0, no)
        depth-4 student: [B1, B2]  where B1: (nh0, nh1), B2: (nh1, no)
    """
    if model_depth == 3:
        if cfg.b1_scale is None:
            raise ValueError("FA model=3 requires b1_scale")
        return [uniform((cfg.nh0, cfg.no), cfg.b1_scale, rng)]
    if model_depth == 4:
        if cfg.b1_scale is None or cfg.b2_scale is None:
            raise ValueError("FA model=4 requires b1_scale and b2_scale")
        return [
            uniform((cfg.nh0, cfg.nh1), cfg.b1_scale, rng),
            uniform((cfg.nh1, cfg.no), cfg.b2_scale, rng),
        ]
    raise ValueError(f"model_depth must be 3 or 4, got {model_depth}")


# --- eval


def _test_nse(student: MLP, Xte: np.ndarray, Yte: np.ndarray, eps: float = 1e-12) -> float:
    """Vectorised mean per-example NSE over the test set."""
    Yhat = student.forward_batch(Xte)
    num = np.sum((Yte - Yhat) ** 2, axis=1)
    den = np.sum(Yte**2, axis=1)
    den = np.maximum(den, eps)
    return float(np.mean(num / den))


# --- main driver


def run_nonlinear(
    cfg: NonlinearConfig,
    model_depth: int,
    alg: Algorithm,
    out_csv: Path,
    *,
    verbose: bool = True,
) -> list[dict]:
    """Train a depth-3 or depth-4 student under BP / FA / shallow and write a CSV."""
    rngs = split_rngs(cfg.seed, ["teacher", "student", "feedback", "test", "stream"])

    teacher = _build_teacher(cfg, rngs["teacher"])
    sizes, acts = _student_arch(model_depth, cfg)
    student = init_mlp(
        sizes=sizes, activations=acts, init_scale=cfg.init_scale, rng=rngs["student"], use_bias=True
    )

    feedback: list[np.ndarray] | None = None
    if alg == "fa":
        feedback = _build_feedback(cfg, model_depth, rngs["feedback"])

    Xte = rngs["test"].normal(0.0, 1.0, size=(cfg.test_size, cfg.ni)).astype(np.float64)
    Yte = teacher.forward_batch(Xte)

    rows: list[dict] = [{"examples_seen": 0, "test_nse": _test_nse(student, Xte, Yte)}]

    iterator = range(cfg.steps)
    if verbose:
        iterator = tqdm(iterator, desc=f"nonlinear-{model_depth}-{alg}")

    stream_rng = rngs["stream"]
    for t in iterator:
        # Stream one example at a time so we never materialise a multi-million-row Xtr.
        x = stream_rng.normal(0.0, 1.0, size=cfg.ni)
        _, posts = teacher.forward(x)
        y_star = posts[-1]
        step(student, x, y_star, alg=alg, eta=cfg.eta, feedback_matrices=feedback)
        if (t + 1) % cfg.eval_every == 0:
            rows.append({"examples_seen": t + 1, "test_nse": _test_nse(student, Xte, Yte)})

    write_rows(out_csv, rows)
    return rows


def nonlinear_sweep(
    cfg: NonlinearConfig,
    b1_list: list[float],
    b2_list: list[float],
    target_scale_list: list[float],
    out_csv: Path,
    *,
    verbose: bool = False,
) -> None:
    """Run the manual-search grid: BP baseline + FA at each (ts, b1, b2)."""
    fieldnames = ["model", "alg", "target_scale", "b1_scale", "b2_scale", "final_test_nse"]
    f, writer = stream_rows(out_csv, fieldnames)
    try:
        for ts in target_scale_list:
            # BP baseline at this teacher regime
            for model in (3, 4):
                tmp = out_csv.parent / f"_tmp_bp_m{model}_ts{ts}.csv"
                cfg_run = NonlinearConfig(**{**cfg.__dict__, "target_scale": ts})
                rows = run_nonlinear(cfg_run, model, "bp", tmp, verbose=verbose)
                writer.writerow(
                    {
                        "model": model,
                        "alg": "bp",
                        "target_scale": ts,
                        "b1_scale": "",
                        "b2_scale": "",
                        "final_test_nse": rows[-1]["test_nse"],
                    }
                )
                tmp.unlink(missing_ok=True)

            # FA model 3 (uses b1 only - no redundant b2 sweep)
            for b1 in b1_list:
                tmp = out_csv.parent / f"_tmp_fa_m3_ts{ts}_b1{b1}.csv"
                cfg_run = NonlinearConfig(**{**cfg.__dict__, "target_scale": ts, "b1_scale": b1})
                rows = run_nonlinear(cfg_run, 3, "fa", tmp, verbose=verbose)
                writer.writerow(
                    {
                        "model": 3,
                        "alg": "fa",
                        "target_scale": ts,
                        "b1_scale": b1,
                        "b2_scale": "",
                        "final_test_nse": rows[-1]["test_nse"],
                    }
                )
                tmp.unlink(missing_ok=True)

            # FA model 4 (uses b1 and b2)
            for b1 in b1_list:
                for b2 in b2_list:
                    tmp = out_csv.parent / f"_tmp_fa_m4_ts{ts}_b1{b1}_b2{b2}.csv"
                    cfg_run = NonlinearConfig(
                        **{**cfg.__dict__, "target_scale": ts, "b1_scale": b1, "b2_scale": b2}
                    )
                    rows = run_nonlinear(cfg_run, 4, "fa", tmp, verbose=verbose)
                    writer.writerow(
                        {
                            "model": 4,
                            "alg": "fa",
                            "target_scale": ts,
                            "b1_scale": b1,
                            "b2_scale": b2,
                            "final_test_nse": rows[-1]["test_nse"],
                        }
                    )
                    tmp.unlink(missing_ok=True)
    finally:
        f.close()
