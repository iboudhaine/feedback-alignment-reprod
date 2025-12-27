import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .utils import uniform_matrix, tanh_prime_from_act, nse, angle

@dataclass
class NonlinearConfig:
    ni: int = 30
    nh0: int = 20
    nh1: int = 10
    no: int = 10
    steps: int = 1500000
    eta: float = 1e-3
    seed: int = 0
    test_size: int = 5000
    target_scale: float = None        # paper: chosen to create regime where deeper BP helps (manual)
    b1_scale: float = None            # paper: chosen manually
    b2_scale: float = None

def _tanh(x): return np.tanh(x)

def make_target(cfg: NonlinearConfig, rng: np.random.Generator):
    if cfg.target_scale is None:
        raise ValueError("Task(3): paper chooses target regime manually; pass --target_scale")
    # Target: y* = W2 tanh(W1 tanh(W0x + b0) + b1) + b2
    W0 = uniform_matrix((cfg.nh0, cfg.ni), cfg.target_scale, rng)
    b0 = uniform_matrix((cfg.nh0,), cfg.target_scale, rng)
    W1 = uniform_matrix((cfg.nh1, cfg.nh0), cfg.target_scale, rng)
    b1 = uniform_matrix((cfg.nh1,), cfg.target_scale, rng)
    W2 = uniform_matrix((cfg.no, cfg.nh1), cfg.target_scale, rng)
    b2 = uniform_matrix((cfg.no,), cfg.target_scale, rng)
    return (W0,b0,W1,b1,W2,b2)

def target_forward(target_params, x):
    W0,b0,W1,b1,W2,b2 = target_params
    h0 = _tanh(W0 @ x + b0)
    h1 = _tanh(W1 @ h0 + b1)
    y  = (W2 @ h1 + b2)
    return y

def init_model_3(cfg: NonlinearConfig, rng: np.random.Generator, init_scale: float):
    # 30–20–10 model: h0=tanh, y=linear
    W0 = uniform_matrix((cfg.nh0, cfg.ni), init_scale, rng)
    b0 = uniform_matrix((cfg.nh0,), init_scale, rng)
    W1 = uniform_matrix((cfg.no, cfg.nh0), init_scale, rng)
    b1 = uniform_matrix((cfg.no,), init_scale, rng)
    return (W0,b0,W1,b1)

def init_model_4(cfg: NonlinearConfig, rng: np.random.Generator, init_scale: float):
    # 30–20–10–10 model: h0=tanh, h1=tanh, y=linear
    W0 = uniform_matrix((cfg.nh0, cfg.ni), init_scale, rng)
    b0 = uniform_matrix((cfg.nh0,), init_scale, rng)
    W1 = uniform_matrix((cfg.nh1, cfg.nh0), init_scale, rng)
    b1 = uniform_matrix((cfg.nh1,), init_scale, rng)
    W2 = uniform_matrix((cfg.no, cfg.nh1), init_scale, rng)
    b2 = uniform_matrix((cfg.no,), init_scale, rng)
    return (W0,b0,W1,b1,W2,b2)

def run_nonlinear(cfg: NonlinearConfig, model_depth: int, alg: str, out_csv: Path, init_scale: float = 0.01):
    """
    model_depth ∈ {3,4}
    alg ∈ {'bp','fa','shallow'}
    - shallow for model_depth=4 means train only last layer (W2,b2)
    - shallow for model_depth=3 means train only last layer (W1,b1)
    """
    if alg not in {"bp","fa","shallow"}:
        raise ValueError("alg must be bp, fa, or shallow")
    if model_depth not in {3,4}:
        raise ValueError("model_depth must be 3 or 4")
    if alg == "fa" and (cfg.b1_scale is None or cfg.b2_scale is None):
        raise ValueError("FA requires --b1_scale and --b2_scale (paper: selected manually)")
    rng = np.random.default_rng(cfg.seed)

    target_params = make_target(cfg, rng)

    # fixed test-set
    Xte = rng.normal(0.0, 1.0, size=(cfg.test_size, cfg.ni)).astype(np.float64)
    Yte = np.stack([target_forward(target_params, Xte[i]) for i in range(cfg.test_size)], axis=0)

    # training sequence fixed: generate Xtr once (same across algorithms) per paper
    Xtr = rng.normal(0.0, 1.0, size=(cfg.steps, cfg.ni)).astype(np.float64)

    # init model + (for FA) backward matrices
    if model_depth == 3:
        W0,b0,W1,b1 = init_model_3(cfg, rng, init_scale)
        if alg == "fa":
            B1 = rng.uniform(-cfg.b1_scale, cfg.b1_scale, size=(cfg.nh0, cfg.no)).astype(np.float64)
        else:
            B1 = None
    else:
        W0,b0,W1,b1,W2,b2 = init_model_4(cfg, rng, init_scale)
        if alg == "fa":
            B2 = rng.uniform(-cfg.b2_scale, cfg.b2_scale, size=(cfg.nh1, cfg.no)).astype(np.float64)
            B1 = rng.uniform(-cfg.b1_scale, cfg.b1_scale, size=(cfg.nh0, cfg.nh1)).astype(np.float64)
        else:
            B2 = None
            B1 = None

    def test_nse():
        if model_depth == 3:
            errs = []
            for i in range(cfg.test_size):
                x = Xte[i]
                y_star = Yte[i]
                h0 = _tanh(W0 @ x + b0)
                y = W1 @ h0 + b1
                errs.append(nse(y_star, y))
            return float(np.mean(errs))
        else:
            errs = []
            for i in range(cfg.test_size):
                x = Xte[i]
                y_star = Yte[i]
                h0 = _tanh(W0 @ x + b0)
                h1 = _tanh(W1 @ h0 + b1)
                y = W2 @ h1 + b2
                errs.append(nse(y_star, y))
            return float(np.mean(errs))

    rows = []
    eval_every = 50000
    rows.append({"examples_seen": 0, "test_nse": test_nse()})

    for t in tqdm(range(cfg.steps), desc=f"nonlinear-{model_depth}-{alg}"):
        x = Xtr[t]
        y_star = target_forward(target_params, x)

        if model_depth == 3:
            h0 = _tanh(W0 @ x + b0)
            y = W1 @ h0 + b1
            e = (y_star - y)

            # output layer (linear)
            if True:
                W1 = W1 + cfg.eta * np.outer(e, h0)
                b1 = b1 + cfg.eta * e

            if alg != "shallow":
                # error signal to layer0 pre-activation (before h0')
                dh0_bp = (W1.T @ e)
                if alg == "bp":
                    dh0 = dh0_bp
                else:
                    dh0 = (B1 @ e)

                # apply derivative for updating W0
                dh0_eff = dh0 * tanh_prime_from_act(h0)
                W0 = W0 + cfg.eta * np.outer(dh0_eff, x)
                b0 = b0 + cfg.eta * dh0_eff

        else:
            h0 = _tanh(W0 @ x + b0)
            h1 = _tanh(W1 @ h0 + b1)
            y = W2 @ h1 + b2
            e = (y_star - y)

            # output layer (linear)
            W2 = W2 + cfg.eta * np.outer(e, h1)
            b2 = b2 + cfg.eta * e

            if alg != "shallow":
                # follow paper's notation: Δh1 is pre-activation signal, derivative applied when updating lower layer
                dh1_bp = (W2.T @ e)      # shape (nh1,)
                if alg == "bp":
                    dh1 = dh1_bp
                else:
                    dh1 = (B2 @ e)

                # update W1 uses (dh1 ◦ h1') h0^T
                dh1_eff = dh1 * tanh_prime_from_act(h1)
                W1 = W1 + cfg.eta * np.outer(dh1_eff, h0)
                b1 = b1 + cfg.eta * dh1_eff

                # propagate to h0: BP uses W1^T( (W2^T e) ◦ h1' ); FA uses B1( (B2 e) ◦ h1' )
                if alg == "bp":
                    dh0 = (W1.T @ dh1_eff)
                else:
                    dh0 = (B1 @ dh1_eff)

                dh0_eff = dh0 * tanh_prime_from_act(h0)
                W0 = W0 + cfg.eta * np.outer(dh0_eff, x)
                b0 = b0 + cfg.eta * dh0_eff

        if (t + 1) % eval_every == 0:
            rows.append({"examples_seen": t + 1, "test_nse": test_nse()})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def nonlinear_sweep(cfg: NonlinearConfig, b1_list, b2_list, target_scale_list, out_csv: Path):
    """
    Writes a table of final test NSE for combinations, to replicate the paper's manual selection.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["model","alg","target_scale","b1_scale","b2_scale","final_test_nse"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

        for ts in target_scale_list:
            # BP 3-layer and 4-layer
            for model in [3,4]:
                cfg_run = NonlinearConfig(**{**cfg.__dict__, "target_scale": ts})
                tmp = out_csv.parent / f"_tmp_nl_{model}_bp_ts{ts}.csv"
                run_nonlinear(cfg_run, model, "bp", tmp)
                last = list(csv.DictReader(tmp.open("r", encoding="utf-8")))[-1]
                w.writerow({"model": model, "alg": "bp", "target_scale": ts, "b1_scale": "", "b2_scale": "", "final_test_nse": last["test_nse"]})
                tmp.unlink(missing_ok=True)

            for b1 in b1_list:
                for b2 in b2_list:
                    for model in [3,4]:
                        cfg_run = NonlinearConfig(**{**cfg.__dict__, "target_scale": ts, "b1_scale": b1, "b2_scale": b2})
                        tmp = out_csv.parent / f"_tmp_nl_{model}_fa_ts{ts}_b1{b1}_b2{b2}.csv"
                        run_nonlinear(cfg_run, model, "fa", tmp)
                        last = list(csv.DictReader(tmp.open("r", encoding="utf-8")))[-1]
                        w.writerow({"model": model, "alg": "fa", "target_scale": ts, "b1_scale": b1, "b2_scale": b2, "final_test_nse": last["test_nse"]})
                        tmp.unlink(missing_ok=True)
