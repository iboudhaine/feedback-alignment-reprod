"""Unit tests for feedback_alignment.core primitives."""
from __future__ import annotations

import math

import numpy as np

from feedback_alignment.core import (
    Activation,
    angle,
    apply_activation,
    derivative_from_post,
    mask_bernoulli,
    nse,
    one_hot,
    sigmoid,
    split_rngs,
    uniform,
)


def test_angle_parallel_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert angle(a, 2 * a) == 0.0


def test_angle_antiparallel_is_zero() -> None:
    # paper uses |a^T b| so antiparallel returns 0, not pi
    a = np.array([1.0, 2.0, 3.0])
    assert angle(a, -a) == 0.0


def test_angle_orthogonal_is_pi_over_2() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert math.isclose(angle(a, b), math.pi / 2, rel_tol=1e-12)


def test_nse_zero_when_equal() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert nse(y, y) == 0.0


def test_nse_one_when_prediction_is_zero() -> None:
    y_star = np.array([1.0, 2.0, 3.0])
    y = np.zeros_like(y_star)
    assert math.isclose(nse(y_star, y), 1.0, rel_tol=1e-12)


def test_sigmoid_extremes_dont_overflow() -> None:
    z = np.array([-1e3, -50.0, 0.0, 50.0, 1e3])
    out = sigmoid(z)
    assert np.all(np.isfinite(out))
    assert math.isclose(out[2], 0.5, rel_tol=1e-12)


def test_derivatives_match_definitions() -> None:
    a_sig = np.array([0.2, 0.7])
    a_tanh = np.array([-0.5, 0.4])
    assert np.allclose(derivative_from_post(Activation.SIGMOID, a_sig), a_sig * (1 - a_sig))
    assert np.allclose(derivative_from_post(Activation.TANH, a_tanh), 1 - a_tanh ** 2)
    assert np.allclose(derivative_from_post(Activation.LINEAR, a_sig), np.ones_like(a_sig))


def test_apply_activation_roundtrip() -> None:
    z = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(apply_activation(Activation.LINEAR, z), z)
    assert np.allclose(apply_activation(Activation.SIGMOID, z), 1.0 / (1.0 + np.exp(-z)))
    assert np.allclose(apply_activation(Activation.TANH, z), np.tanh(z))


def test_split_rngs_are_independent() -> None:
    rngs = split_rngs(123, ["data", "init"])
    a = rngs["data"].normal(size=1000)
    b = rngs["init"].normal(size=1000)
    # independent streams should not produce identical sequences
    assert not np.allclose(a, b)


def test_split_rngs_reproducible() -> None:
    a = split_rngs(7, ["x"])["x"].normal(size=10)
    b = split_rngs(7, ["x"])["x"].normal(size=10)
    assert np.allclose(a, b)


def test_uniform_in_range() -> None:
    rng = np.random.default_rng(0)
    M = uniform((50, 50), 0.3, rng)
    assert M.min() >= -0.3
    assert M.max() <= 0.3


def test_one_hot() -> None:
    v = one_hot(3, 10)
    assert v.shape == (10,)
    assert v[3] == 1.0
    assert v.sum() == 1.0


def test_mask_bernoulli_keeps_roughly_half() -> None:
    rng = np.random.default_rng(0)
    m = mask_bernoulli((1000, 1000), 0.5, rng)
    fraction = float(m.mean())
    assert 0.49 < fraction < 0.51
