"""One-example online SGD update under BP, feedback alignment, or shallow learning.

This is the single place where the three learning rules are defined. The three
task drivers (linear / MNIST / nonlinear) each call `step()` in their inner loop
instead of re-deriving the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .core import derivative_from_post, nse
from .layers import MLP

Algorithm = Literal["bp", "fa", "shallow"]


@dataclass
class StepResult:
    """Outputs of one online SGD step on a single example.

    y: network output for this example.
    nse: normalized squared error of this example.
    signals_bp: error signal at every layer under standard backprop. Each entry
        already includes the local activation derivative, so it is the
        pre-activation gradient sign-flipped (equal to -dL/dz_i).
    signals_fa: same shape as signals_bp but propagated through the random
        feedback matrices. None when no feedback matrices were supplied.
    posts: per-layer post-activations from the forward pass.
    """

    y: np.ndarray
    nse: float
    signals_bp: list[np.ndarray]
    signals_fa: list[np.ndarray] | None
    posts: list[np.ndarray]


def step(
    mlp: MLP,
    x: np.ndarray,
    y_star: np.ndarray,
    *,
    alg: Algorithm,
    eta: float,
    feedback_matrices: list[np.ndarray] | None = None,
    weight_decay: float = 0.0,
    w_masks: list[np.ndarray | None] | None = None,
) -> StepResult:
    """Forward, compute BP and FA signals, apply the update, return signals and metrics.

    Args:
        mlp: model. Mutated in place.
        x, y_star: single example and its target.
        alg: which signal drives the weight update.
        eta: learning rate.
        feedback_matrices: list of length depth-1. feedback_matrices[i] has
            shape (layers[i].n_out, layers[i+1].n_out) and replaces
            layers[i+1].W.T in the FA error path. Required when alg='fa'.
            Optional otherwise: pass it on a BP run to also log the FA-vs-BP
            angle for diagnostics.
        weight_decay: applied multiplicatively to W only (not biases).
        w_masks: per-layer binary mask reapplied after each W update for sparsity.
    """
    depth = mlp.depth

    if feedback_matrices is not None and len(feedback_matrices) != depth - 1:
        raise ValueError(
            f"feedback_matrices must have length depth-1={depth - 1}, got {len(feedback_matrices)}"
        )
    if alg == "fa" and feedback_matrices is None:
        raise ValueError("alg='fa' requires feedback_matrices")
    if w_masks is not None and len(w_masks) != depth:
        raise ValueError(f"w_masks must have length depth={depth}, got {len(w_masks)}")

    # Forward pass on the pre-update weights.
    pres, posts = mlp.forward(x)
    y = posts[-1]

    # Top-layer signal for the loss L = 0.5 * ||y_star - y||^2.
    e = y_star - y
    top_signal = e * derivative_from_post(mlp.layers[-1].activation, y)

    # BP signals at every layer. Always computed so we can report the
    # BP-vs-FA angle on FA runs.
    signals_bp: list[np.ndarray] = [np.empty(0)] * depth
    signals_bp[-1] = top_signal
    for i in range(depth - 2, -1, -1):
        upstream = mlp.layers[i + 1].W.T @ signals_bp[i + 1]
        signals_bp[i] = upstream * derivative_from_post(mlp.layers[i].activation, posts[i])

    # FA signals cascade through the fixed random feedback matrices.
    signals_fa: list[np.ndarray] | None = None
    if feedback_matrices is not None:
        signals_fa = [np.empty(0)] * depth
        signals_fa[-1] = top_signal
        for i in range(depth - 2, -1, -1):
            upstream = feedback_matrices[i] @ signals_fa[i + 1]
            signals_fa[i] = upstream * derivative_from_post(mlp.layers[i].activation, posts[i])

    # Choose update direction and which layers train.
    if alg == "shallow":
        layers_to_update = [depth - 1]
        signals_used = signals_bp
    elif alg == "bp":
        layers_to_update = list(range(depth))
        signals_used = signals_bp
    else:  # fa
        layers_to_update = list(range(depth))
        assert signals_fa is not None  # narrowed by validation above
        signals_used = signals_fa

    # Apply the per-layer updates.
    for i in layers_to_update:
        signal = signals_used[i]
        post_below = posts[i - 1] if i > 0 else x
        layer = mlp.layers[i]
        if weight_decay:
            layer.W *= 1.0 - weight_decay
        layer.W += eta * np.outer(signal, post_below)
        if layer.use_bias:
            layer.b += eta * signal
        if w_masks is not None and w_masks[i] is not None:
            layer.W *= w_masks[i]

    return StepResult(
        y=y,
        nse=nse(y_star, y),
        signals_bp=signals_bp,
        signals_fa=signals_fa,
        posts=posts,
    )
