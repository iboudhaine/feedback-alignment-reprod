import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .utils import sigmoid, sigmoid_prime_from_act, one_hot, uniform_matrix, angle, mask50

try:
    import torch
    from torchvision import datasets, transforms
except Exception as _e:
    torch = None
    datasets = None
    transforms = None

@dataclass
class MNISTConfig:
    ni: int = 784
    nh: int = 1000
    no: int = 10
    eta: float = 1e-3            # paper fixed
    alpha: float = 1e-6          # weight decay
    omega: float = None          # manual search in paper
    beta: float = None           # manual search in paper
    max_examples: int = 1500000  # Fig.2 axis to 15×10^5
    eval_every: int = 50000
    seed: int = 0
    sparse50: bool = False

def load_mnist(data_dir: Path):
    if torch is None:
        raise RuntimeError("torch/torchvision are required for MNIST loading")
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=tfm)
    return train, test

def _prep_dataset(ds):
    # ds[i] -> (1x28x28 tensor, label)
    X = np.zeros((len(ds), 784), dtype=np.float64)
    y = np.zeros((len(ds),), dtype=np.int64)
    for i in range(len(ds)):
        img, lab = ds[i]
        X[i] = np.asarray(img, dtype=np.float64).reshape(-1)  # in [0,1]
        y[i] = int(lab)
    return X, y

def _init_params(cfg: MNISTConfig, rng: np.random.Generator):
    if cfg.omega is None:
        raise ValueError("MNIST: paper selects ω by manual search; pass --omega")
    W0 = uniform_matrix((cfg.nh, cfg.ni), cfg.omega, rng)
    b0 = uniform_matrix((cfg.nh,), cfg.omega, rng)
    W  = uniform_matrix((cfg.no, cfg.nh), cfg.omega, rng)
    b  = uniform_matrix((cfg.no,), cfg.omega, rng)

    if cfg.beta is None:
        B = None
    else:
        B = rng.uniform(-cfg.beta, cfg.beta, size=(cfg.nh, cfg.no)).astype(np.float64)

    if cfg.sparse50:
        mask_W = mask50(W.shape, rng)
        W *= mask_W
        if B is not None:
            mask_B = mask50(B.shape, rng)
            B *= mask_B
        else:
            mask_B = None
    else:
        mask_W = None
        mask_B = None

    return W0, b0, W, b, B, mask_W, mask_B

def _forward(W0, b0, W, b, x):
    h_net = W0 @ x + b0
    h = sigmoid(h_net)
    y_net = W @ h + b
    y = sigmoid(y_net)
    return h, y

def _test_error(W0, b0, W, b, Xtest, ytest):
    errs = 0
    for i in range(len(Xtest)):
        h, y = _forward(W0, b0, W, b, Xtest[i])
        pred = int(np.argmax(y))
        errs += int(pred != int(ytest[i]))
    return 100.0 * errs / len(Xtest)

def run_mnist(cfg: MNISTConfig, alg: str, out_csv: Path, data_dir: Path):
    """
    alg ∈ {'bp','fa','shallow'}
    - bp: backprop
    - fa: feedback alignment (requires beta, B matrix)
    - shallow: train only output layer (W,b); freeze W0,b0 (paper's "shallow learning")
    """
    if alg not in {"bp","fa","shallow"}:
        raise ValueError("alg must be one of: bp, fa, shallow")
    if alg == "fa" and cfg.beta is None:
        raise ValueError("FA requires --beta (paper: B ~ Uniform[-β,β])")

    train_ds, test_ds = load_mnist(data_dir)
    Xtr, ytr = _prep_dataset(train_ds)
    Xte, yte = _prep_dataset(test_ds)

    rng = np.random.default_rng(cfg.seed)
    W0, b0, W, b, B, mask_W, mask_B = _init_params(cfg, rng)

    # Fixed training sequence across algorithms: use a deterministic permutation stream
    idx = np.arange(len(Xtr))
    rng.shuffle(idx)

    rows = []
    # initial test
    rows.append({"examples_seen": 0, "test_error_pct": _test_error(W0, b0, W, b, Xte, yte), "mean_angle_dh_fa_bp": np.nan})

    # moving average accumulator for angle (as in paper: time-averaged mean)
    ang_sum = 0.0
    ang_count = 0

    for t in tqdm(range(cfg.max_examples), desc=f"mnist-{alg}"):
        j = idx[t % len(idx)]
        x = Xtr[j]
        y_star = one_hot(int(ytr[j]), 10)

        h, y = _forward(W0, b0, W, b, x)
        e = (y_star - y)

        # standard squared-error with sigmoid output: include output derivative in weight-gradient
        dy = e * sigmoid_prime_from_act(y)

        # output layer update
        if alg in {"bp","fa","shallow"}:
            # weight decay on synaptic weights (paper: α = 1e-6)
            W = (1.0 - cfg.alpha) * W + cfg.eta * np.outer(dy, h)
            b = b + cfg.eta * dy

            if mask_W is not None:
                W *= mask_W

        if alg != "shallow":
            # hidden update signals (for angle measurement use FA vs BP)
            dh_bp = (W.T @ dy)
            dh_fa = (B @ dy) if B is not None else None

            # multiply by hidden derivative for weight updates into hidden layer
            dh_bp_eff = dh_bp * sigmoid_prime_from_act(h)
            dh_fa_eff = dh_fa * sigmoid_prime_from_act(h) if dh_fa is not None else None

            if alg == "bp":
                dh_eff = dh_bp_eff
            elif alg == "fa":
                dh_eff = dh_fa_eff
            else:
                raise AssertionError("unreachable")

            W0 = (1.0 - cfg.alpha) * W0 + cfg.eta * np.outer(dh_eff, x)
            b0 = b0 + cfg.eta * dh_eff

            if mask_B is not None:
                B *= mask_B  # keep sparsity fixed
            # keep W0 unmasked (paper sparsity described for W and B only)

            # angle statistics (paper Fig 2b)
            if dh_fa_eff is not None:
                ang_sum += angle(dh_fa_eff, dh_bp_eff)
                ang_count += 1

        if (t + 1) % cfg.eval_every == 0:
            mean_ang = (ang_sum / ang_count) if ang_count > 0 else np.nan
            rows.append({
                "examples_seen": t + 1,
                "test_error_pct": _test_error(W0, b0, W, b, Xte, yte),
                "mean_angle_dh_fa_bp": mean_ang,
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def run_mnist_sweep(cfg: MNISTConfig, omega_list, beta_list, out_csv: Path, data_dir: Path):
    """
    Implements the paper's "manual search" by producing results for a grid of (ω,β).
    You choose the final (ω,β) by inspecting final test error.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["omega","beta","final_test_error_pct"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for omega in omega_list:
            # backprop run to evaluate ω
            cfg_bp = MNISTConfig(**{**cfg.__dict__, "omega": omega, "beta": None})
            tmp = Path(out_csv.parent) / f"_tmp_bp_omega{omega}.csv"
            run_mnist(cfg_bp, "bp", tmp, data_dir)
            # take last line
            last = list(csv.DictReader(tmp.open("r", encoding="utf-8")))[-1]
            w.writerow({"omega": omega, "beta": "", "final_test_error_pct": last["test_error_pct"]})
            tmp.unlink(missing_ok=True)

            # for each β run FA
            for beta in beta_list:
                cfg_fa = MNISTConfig(**{**cfg.__dict__, "omega": omega, "beta": beta})
                tmp = Path(out_csv.parent) / f"_tmp_fa_omega{omega}_beta{beta}.csv"
                run_mnist(cfg_fa, "fa", tmp, data_dir)
                last = list(csv.DictReader(tmp.open("r", encoding="utf-8")))[-1]
                w.writerow({"omega": omega, "beta": beta, "final_test_error_pct": last["test_error_pct"]})
                tmp.unlink(missing_ok=True)
