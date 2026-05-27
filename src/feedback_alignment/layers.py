"""Tiny MLP abstraction used by all three tasks.

A Layer holds (W, b, activation). An MLP is a list of layers. Forward returns the
pre- and post-activation of every layer so both BP and FA can read what they need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Activation, apply_activation, uniform


@dataclass
class Layer:
    W: np.ndarray  # (n_out, n_in)
    b: np.ndarray  # (n_out,). Zeros when use_bias=False; never trained in that case.
    activation: Activation
    use_bias: bool = True

    @property
    def n_in(self) -> int:
        return int(self.W.shape[1])

    @property
    def n_out(self) -> int:
        return int(self.W.shape[0])


@dataclass
class MLP:
    layers: list[Layer]

    @property
    def depth(self) -> int:
        return len(self.layers)

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Forward pass on a single example.

        Returns (pre_activations, post_activations) - both length `depth`.
        `post_activations[-1]` is the network output.
        """
        pres: list[np.ndarray] = []
        posts: list[np.ndarray] = []
        a = x
        for layer in self.layers:
            z = layer.W @ a + layer.b
            a = apply_activation(layer.activation, z)
            pres.append(z)
            posts.append(a)
        return pres, posts

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised forward over a batch X of shape (N, n_in). Returns network output."""
        A = X
        for layer in self.layers:
            Z = A @ layer.W.T + layer.b
            A = apply_activation(layer.activation, Z)
        return A


def init_mlp(
    sizes: list[int],
    activations: list[Activation],
    init_scale: float | list[float],
    rng: np.random.Generator,
    use_bias: bool = True,
    bias_scale: float | None = None,
) -> MLP:
    """Build an MLP with uniform[-scale, scale] weights.

    `sizes` has length depth+1 (input size followed by each layer's output size).
    `init_scale` is either a scalar (same scale every layer) or a list of length
    depth giving one scale per layer - useful for teacher networks whose regime
    differs across layers.

    When `use_bias=False`, biases are zeros and never updated. Otherwise biases
    are drawn uniform[-bias_scale, bias_scale]; if `bias_scale` is None, the
    per-layer weight scale is reused.
    """
    if len(activations) != len(sizes) - 1:
        raise ValueError("len(activations) must equal len(sizes) - 1")
    depth = len(activations)
    scales = (
        [float(init_scale)] * depth if isinstance(init_scale, (int, float)) else list(init_scale)
    )
    if len(scales) != depth:
        raise ValueError(f"init_scale list must have length {depth}, got {len(scales)}")

    layers: list[Layer] = []
    for n_in, n_out, act, scale in zip(sizes[:-1], sizes[1:], activations, scales, strict=True):
        W = uniform((n_out, n_in), scale, rng)
        if use_bias:
            bs = scale if bias_scale is None else bias_scale
            b = uniform((n_out,), bs, rng)
        else:
            b = np.zeros((n_out,))
        layers.append(Layer(W=W, b=b, activation=act, use_bias=use_bias))
    return MLP(layers=layers)
