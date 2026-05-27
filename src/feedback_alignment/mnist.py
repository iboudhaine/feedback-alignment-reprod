"""Task 2: MNIST classification with feedback alignment.

Architecture: 784 -> 1000 -> 10, sigmoid hidden and sigmoid output, MSE loss
with weight decay alpha and optional 50% sparsity masks on W and B. The paper
specifies eta = 1e-3 and alpha = 1e-6; omega (init scale) and beta (feedback
scale) are chosen by manual search via `run_mnist_sweep`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .algorithms import Algorithm, step
from .core import Activation, angle, mask_bernoulli, one_hot, split_rngs, uniform
from .io import build_slug, stream_rows, write_rows
from .layers import MLP, init_mlp


@dataclass
class MNISTConfig:
    ni: int = 784
    nh: int = 1000
    no: int = 10
    eta: float = 1e-3
    alpha: float = 1e-6                 # weight decay (paper)
    omega: float | None = None          # W init scale (manual search)
    beta: float | None = None           # B feedback scale (manual search)
    max_examples: int = 1_500_000
    eval_every: int = 50_000
    seed: int = 0
    sparse50: bool = False              # mask 50% of W (output) and B at init

    def slug(self) -> str:
        return build_slug([
            ("seed", self.seed),
            ("omega", self.omega),
            ("beta", self.beta),
            ("sparse50", self.sparse50),
        ])


def load_mnist(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST and return numpy arrays.

    Uses torchvision under the hood, then converts the raw uint8 tensors to
    numpy in one shot rather than iterating example by example.
    """
    try:
        from torchvision import datasets  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError("torchvision is required to load MNIST") from e

    data_dir.mkdir(parents=True, exist_ok=True)
    train = datasets.MNIST(root=str(data_dir), train=True, download=True)
    test = datasets.MNIST(root=str(data_dir), train=False, download=True)
    x_train = train.data.numpy().astype(np.float64).reshape(-1, 784) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    x_test = test.data.numpy().astype(np.float64).reshape(-1, 784) / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    return x_train, y_train, x_test, y_test


def test_error_pct(mlp: MLP, X: np.ndarray, y: np.ndarray) -> float:
    """Vectorised classification-error percentage on a labelled set."""
    Yhat = mlp.forward_batch(X)
    preds = np.argmax(Yhat, axis=1)
    return 100.0 * float((preds != y).mean())


def run_mnist(
    cfg: MNISTConfig,
    alg: Algorithm,
    out_csv: Path,
    data_dir: Path,
    *,
    verbose: bool = True,
) -> list[dict]:
    """Train MNIST under BP / FA / shallow, logging test error and BP-vs-FA angle."""
    if cfg.omega is None:
        raise ValueError("omega must be set (paper: manual search; see run_mnist_sweep)")
    if alg == "fa" and cfg.beta is None:
        raise ValueError("alg='fa' requires beta (paper: B ~ Uniform[-beta, beta])")

    rngs = split_rngs(cfg.seed, ["init", "feedback", "mask_w", "mask_b", "order"])

    x_train, y_train, x_test, y_test = load_mnist(data_dir)

    student = init_mlp(
        sizes=[cfg.ni, cfg.nh, cfg.no],
        activations=[Activation.SIGMOID, Activation.SIGMOID],
        init_scale=cfg.omega,
        rng=rngs["init"],
        use_bias=True,
    )

    # Feedback matrix. Built when beta is provided. Required for FA; optional
    # otherwise so BP runs can also report the FA-vs-BP angle for diagnostics.
    feedback: list[np.ndarray] | None = None
    if cfg.beta is not None:
        B = uniform((cfg.nh, cfg.no), cfg.beta, rngs["feedback"])
        if cfg.sparse50:
            # B is fixed throughout training, so the mask is applied once at init.
            B *= mask_bernoulli(B.shape, 0.5, rngs["mask_b"])
        feedback = [B]

    # The output-layer W mask is reapplied by step() after every update.
    # W0 is intentionally left dense (the paper masks W and B only).
    w_masks: list[np.ndarray | None] | None = None
    if cfg.sparse50:
        mask_W_out = mask_bernoulli(student.layers[1].W.shape, 0.5, rngs["mask_w"])
        student.layers[1].W *= mask_W_out
        w_masks = [None, mask_W_out]

    rows: list[dict] = [{
        "examples_seen": 0,
        "test_error_pct": test_error_pct(student, x_test, y_test),
        "mean_angle_delta_h_fa_bp": float("nan"),
    }]

    ang_sum = 0.0
    ang_count = 0
    examples_seen = 0
    n_train = len(x_train)

    pbar = tqdm(total=cfg.max_examples, desc=f"mnist-{alg}", disable=not verbose)
    while examples_seen < cfg.max_examples:
        # Reshuffle each epoch rather than cycling a single fixed permutation.
        perm = rngs["order"].permutation(n_train)
        for idx in perm:
            x = x_train[idx]
            y_star = one_hot(int(y_train[idx]), cfg.no)

            result = step(
                student, x, y_star,
                alg=alg,
                eta=cfg.eta,
                feedback_matrices=feedback,
                weight_decay=cfg.alpha,
                w_masks=w_masks,
            )

            if result.signals_fa is not None and alg != "shallow":
                ang_sum += angle(result.signals_fa[0], result.signals_bp[0])
                ang_count += 1

            examples_seen += 1
            pbar.update(1)

            if examples_seen % cfg.eval_every == 0:
                mean_ang = ang_sum / ang_count if ang_count else float("nan")
                rows.append({
                    "examples_seen": examples_seen,
                    "test_error_pct": test_error_pct(student, x_test, y_test),
                    "mean_angle_delta_h_fa_bp": mean_ang,
                })

            if examples_seen >= cfg.max_examples:
                break
    pbar.close()

    write_rows(out_csv, rows)
    return rows


def run_mnist_sweep(
    cfg: MNISTConfig,
    omega_list: list[float],
    beta_list: list[float],
    out_csv: Path,
    data_dir: Path,
    *,
    include_bp_baseline: bool = True,
    verbose: bool = False,
) -> None:
    """Manual-search grid over (omega, beta).

    Writes one row per (omega, beta) FA combination. The BP baseline at each
    omega is included for reference (one row per omega) unless
    include_bp_baseline is False.
    """
    fieldnames = ["alg", "omega", "beta", "final_test_error_pct"]
    f, writer = stream_rows(out_csv, fieldnames)
    try:
        for omega in omega_list:
            if include_bp_baseline:
                tmp = out_csv.parent / f"_tmp_bp_omega{omega}.csv"
                cfg_bp = replace(cfg, omega=omega, beta=None)
                rows = run_mnist(cfg_bp, "bp", tmp, data_dir, verbose=verbose)
                writer.writerow({
                    "alg": "bp", "omega": omega, "beta": "",
                    "final_test_error_pct": rows[-1]["test_error_pct"],
                })
                tmp.unlink(missing_ok=True)

            for beta in beta_list:
                tmp = out_csv.parent / f"_tmp_fa_omega{omega}_beta{beta}.csv"
                cfg_fa = replace(cfg, omega=omega, beta=beta)
                rows = run_mnist(cfg_fa, "fa", tmp, data_dir, verbose=verbose)
                writer.writerow({
                    "alg": "fa", "omega": omega, "beta": beta,
                    "final_test_error_pct": rows[-1]["test_error_pct"],
                })
                tmp.unlink(missing_ok=True)
    finally:
        f.close()
