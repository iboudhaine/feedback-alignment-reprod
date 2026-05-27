"""Math primitives, RNG helpers, and the activation enum used across all tasks."""
from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

import numpy as np

# ---------- activations ----------

class Activation(str, Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    TANH = "tanh"


def apply_activation(act: Activation, z: np.ndarray) -> np.ndarray:
    if act is Activation.LINEAR:
        return z
    if act is Activation.SIGMOID:
        return sigmoid(z)
    if act is Activation.TANH:
        return np.tanh(z)
    raise ValueError(f"Unknown activation: {act}")


def derivative_from_post(act: Activation, post: np.ndarray) -> np.ndarray:
    """Activation derivative expressed in terms of the post-activation value.

    For linear: 1. For sigmoid with a = sigmoid(z): a * (1 - a). For tanh: 1 - a**2.
    """
    if act is Activation.LINEAR:
        return np.ones_like(post)
    if act is Activation.SIGMOID:
        return post * (1.0 - post)
    if act is Activation.TANH:
        return 1.0 - post * post
    raise ValueError(f"Unknown activation: {act}")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[neg])
    out[neg] = expz / (1.0 + expz)
    return out


# ---------- metrics ----------

def angle(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Angle between two vectors, restricted to [0, pi/2].

    Uses the absolute inner product so antiparallel vectors return 0. This
    matches the paper's convention.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = max(eps, float(np.linalg.norm(a) * np.linalg.norm(b)))
    cosv = abs(float(np.dot(a, b))) / denom
    cosv = min(1.0, max(0.0, cosv))
    return float(math.acos(cosv))


def nse(y_star: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized squared error: ||y_star - y||^2 / ||y_star||^2."""
    y_star = np.asarray(y_star, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    diff = y_star - y
    num = float(np.dot(diff, diff))
    den = float(np.dot(y_star, y_star))
    return num / max(eps, den)


# ---------- rng ----------

def split_rngs(seed: int, names: Sequence[str]) -> dict[str, np.random.Generator]:
    """Return one independent Generator per name, all derived from `seed`.

    Each Generator's stream is statistically independent of the others, so data
    sampling cannot accidentally correlate with weight initialisation or
    example ordering.
    """
    seeds = np.random.SeedSequence(seed).spawn(len(names))
    return {name: np.random.default_rng(s) for name, s in zip(names, seeds, strict=True)}


# ---------- init helpers ----------

def uniform(shape: tuple[int, ...] | int, scale: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(low=-scale, high=scale, size=shape).astype(np.float64)


def one_hot(label: int, n_classes: int) -> np.ndarray:
    v = np.zeros((n_classes,), dtype=np.float64)
    v[int(label)] = 1.0
    return v


def mask_bernoulli(shape: tuple[int, ...], keep_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Binary mask where each entry is kept (1.0) with probability `keep_prob`."""
    return (rng.random(shape) < keep_prob).astype(np.float64)
