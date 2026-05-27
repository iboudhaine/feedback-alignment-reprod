"""Gradient check: our hand-coded BP signals must match PyTorch autograd.

This is the load-bearing correctness test for the whole repo. If anyone ever
flips a sign or drops an activation derivative in `algorithms.step`, this fails.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from feedback_alignment.algorithms import step
from feedback_alignment.core import Activation
from feedback_alignment.layers import init_mlp


def _build_pair(sizes: list[int], acts: list[Activation], rng: np.random.Generator):
    """Construct a numpy MLP and a PyTorch model with the SAME parameters."""
    mlp = init_mlp(sizes=sizes, activations=acts, init_scale=0.5, rng=rng, use_bias=True)

    layers: list[nn.Module] = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True):
        layers.append(nn.Linear(n_in, n_out).double())

    with torch.no_grad():
        for i, lin in enumerate(layers):
            assert isinstance(lin, nn.Linear)
            lin.weight.copy_(torch.from_numpy(mlp.layers[i].W))
            lin.bias.copy_(torch.from_numpy(mlp.layers[i].b))

    def torch_forward(x_t: torch.Tensor) -> torch.Tensor:
        a = x_t
        for i, lin in enumerate(layers):
            z = lin(a)
            if acts[i] is Activation.LINEAR:
                a = z
            elif acts[i] is Activation.TANH:
                a = torch.tanh(z)
            elif acts[i] is Activation.SIGMOID:
                a = torch.sigmoid(z)
        return a

    return mlp, layers, torch_forward


def _check_against_autograd(sizes: list[int], acts: list[Activation], seed: int) -> None:
    rng = np.random.default_rng(seed)
    mlp, torch_layers, torch_forward = _build_pair(sizes, acts, rng)

    x = rng.normal(size=sizes[0])
    y_star = rng.uniform(-1.0, 1.0, size=sizes[-1])

    # --- autograd reference
    xt = torch.from_numpy(x).double()
    yt_star = torch.from_numpy(y_star).double()
    yhat = torch_forward(xt)
    loss = 0.5 * ((yt_star - yhat) ** 2).sum()
    loss.backward()

    # Hand-coded BP signals. eta=0 means no weight mutation.
    result = step(mlp, x, y_star, alg="bp", eta=0.0)

    # signal_i = -dL/dz_i, so dL/dW_i = -outer(signal_i, post_{i-1}) and dL/db_i = -signal_i
    depth = mlp.depth
    for i in range(depth):
        post_below = result.posts[i - 1] if i > 0 else x
        expected_dW = -np.outer(result.signals_bp[i], post_below)
        expected_db = -result.signals_bp[i]

        lin = torch_layers[i]
        assert isinstance(lin, nn.Linear)
        autograd_dW = lin.weight.grad.detach().numpy()
        autograd_db = lin.bias.grad.detach().numpy()

        assert np.allclose(autograd_dW, expected_dW, atol=1e-10), (
            f"Layer {i} weight gradient mismatch: max diff = "
            f"{np.max(np.abs(autograd_dW - expected_dW))}"
        )
        assert np.allclose(autograd_db, expected_db, atol=1e-10), (
            f"Layer {i} bias gradient mismatch: max diff = "
            f"{np.max(np.abs(autograd_db - expected_db))}"
        )


def test_grad_check_linear_two_layer() -> None:
    _check_against_autograd([3, 2, 2], [Activation.LINEAR, Activation.LINEAR], seed=0)


def test_grad_check_tanh_sigmoid_two_layer() -> None:
    _check_against_autograd([4, 3, 2], [Activation.TANH, Activation.SIGMOID], seed=1)


def test_grad_check_deep_tanh() -> None:
    _check_against_autograd([5, 4, 3, 2], [Activation.TANH, Activation.TANH, Activation.LINEAR], seed=2)


def test_grad_check_mnist_arch() -> None:
    """Smaller version of the MNIST 784-1000-10 sigmoid arch."""
    _check_against_autograd([8, 6, 4], [Activation.SIGMOID, Activation.SIGMOID], seed=3)
