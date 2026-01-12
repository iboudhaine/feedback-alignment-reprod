import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .utils import set_seed, uniform_matrix, nse, angle

@dataclass
class LinearTaskConfig:
    ni: int = 30
    nh: int = 20
    no: int = 10
    n_examples: int = 2000
    eta: float = 1e-3
    omega: float = 0.01            # W0,W init scale; Task (1) specifies [-0.01,0.01]
    b_scale: float = 0.5           # Task (1) specifies B ~ Uniform[-0.5,0.5]
    seed: int = 0

def generate_task1_dataset(cfg: LinearTaskConfig):
    rng = np.random.default_rng(cfg.seed)
    T = rng.uniform(-1.0, 1.0, size=(cfg.no, cfg.ni)).astype(np.float64)
    X = rng.normal(0.0, 1.0, size=(cfg.n_examples, cfg.ni)).astype(np.float64)
    Y = (X @ T.T).astype(np.float64)
    return T, X, Y

def run_linear_once(cfg: LinearTaskConfig, alg: str, X, Y):
    """
    alg ∈ {'bp','fa'}
    """
    rng = np.random.default_rng(cfg.seed)
    W0 = uniform_matrix((cfg.nh, cfg.ni), cfg.omega, rng)
    W  = uniform_matrix((cfg.no, cfg.nh), cfg.omega, rng)
    B  = rng.uniform(-cfg.b_scale, cfg.b_scale, size=(cfg.nh, cfg.no)).astype(np.float64)

    rows = []
    for t in range(cfg.n_examples):
        x = X[t]
        y_star = Y[t]

        h = W0 @ x
        y = W @ h
        e = (y_star - y)

        dh_bp = W.T @ e
        dh_fa = B @ e

        if alg == "bp":
            dh = dh_bp
        elif alg == "fa":
            dh = dh_fa
        else:
            raise ValueError("alg must be 'bp' or 'fa'")

        # gradient descent on 0.5||e||^2 => update W += η e h^T, W0 += η dh x^T
        W  = W  + cfg.eta * np.outer(e, h)
        W0 = W0 + cfg.eta * np.outer(dh, x)

        rows.append({
            "step": t,
            "nse": nse(y_star, y),
            "angle_dh_fa_bp": angle(dh_fa, dh_bp),
        })
    return rows

def run_fig1(cfg: LinearTaskConfig, alg: str, out_csv: Path):
    T, X, Y = generate_task1_dataset(cfg)
    rows = run_linear_once(cfg, alg, X, Y)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def run_fig4(omega_list, n_trials: int, eta: float, seed: int, out_csv: Path):
    """
    Fig. 4 uses same setup as Fig. 1, but:
      - learning rate eta = 1e-3
      - vary omega (init scale of W0,W)
    We report per-step:
      - NSE
      - angle(dh_fa, dh_bp)
      - angle(dh_fa, dh_pbp), where dh_pbp = W^+ e
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = ["omega","trial","step","nse","angle_dh_fa_bp","angle_dh_fa_pbp"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

        for omega in omega_list:
            for trial in range(n_trials):
                cfg = LinearTaskConfig(eta=eta, omega=omega, seed=seed + 1000*trial, n_examples=2000)
                _, X, Y = generate_task1_dataset(cfg)

                rng = np.random.default_rng(cfg.seed)
                W0 = uniform_matrix((cfg.nh, cfg.ni), cfg.omega, rng)
                W  = uniform_matrix((cfg.no, cfg.nh), cfg.omega, rng)
                B  = rng.uniform(-cfg.b_scale, cfg.b_scale, size=(cfg.nh, cfg.no)).astype(np.float64)

                for t in range(cfg.n_examples):
                    x = X[t]
                    y_star = Y[t]
                    h = W0 @ x
                    y = W @ h
                    e = (y_star - y)

                    dh_bp = W.T @ e
                    dh_fa = B @ e
                    # pseudobackprop: dh_pbp = W^+ e (Moore-Penrose pseudoinverse)
                    W_plus = np.linalg.pinv(W)  # shape (nh, no)
                    dh_pbp = W_plus @ e

                    w.writerow({
                        "omega": omega,
                        "trial": trial,
                        "step": t,
                        "nse": nse(y_star, y),
                        "angle_dh_fa_bp": angle(dh_fa, dh_bp),
                        "angle_dh_fa_pbp": angle(dh_fa, dh_pbp),
                    })

                    # FA learning
                    W  = W  + cfg.eta * np.outer(e, h)
                    W0 = W0 + cfg.eta * np.outer(dh_fa, x)
