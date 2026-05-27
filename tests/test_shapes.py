"""Shape-contract tests for init_mlp / forward / step."""
from __future__ import annotations

import numpy as np

from feedback_alignment.algorithms import step
from feedback_alignment.core import Activation, uniform
from feedback_alignment.layers import init_mlp


def test_init_mlp_layer_shapes() -> None:
    rng = np.random.default_rng(0)
    mlp = init_mlp([30, 20, 10], [Activation.TANH, Activation.LINEAR], 0.01, rng, use_bias=True)
    assert mlp.depth == 2
    assert mlp.layers[0].W.shape == (20, 30)
    assert mlp.layers[0].b.shape == (20,)
    assert mlp.layers[1].W.shape == (10, 20)
    assert mlp.layers[1].b.shape == (10,)


def test_init_mlp_use_bias_false_zeros_biases() -> None:
    rng = np.random.default_rng(0)
    mlp = init_mlp([5, 3, 2], [Activation.LINEAR, Activation.LINEAR], 0.5, rng, use_bias=False)
    for layer in mlp.layers:
        assert np.all(layer.b == 0.0)
        assert layer.use_bias is False


def test_init_mlp_per_layer_init_scale() -> None:
    rng = np.random.default_rng(0)
    mlp = init_mlp(
        sizes=[10, 8, 6, 4],
        activations=[Activation.TANH, Activation.TANH, Activation.LINEAR],
        init_scale=[0.1, 0.5, 1.0],
        rng=rng,
        use_bias=False,
    )
    assert mlp.layers[0].W.max() <= 0.1 and mlp.layers[0].W.min() >= -0.1
    assert mlp.layers[1].W.max() <= 0.5 and mlp.layers[1].W.min() >= -0.5
    assert mlp.layers[2].W.max() <= 1.0 and mlp.layers[2].W.min() >= -1.0


def test_forward_single_example_shapes() -> None:
    rng = np.random.default_rng(0)
    mlp = init_mlp([30, 20, 10], [Activation.TANH, Activation.LINEAR], 0.01, rng)
    x = rng.normal(size=30)
    pres, posts = mlp.forward(x)
    assert len(pres) == 2 and len(posts) == 2
    assert pres[0].shape == posts[0].shape == (20,)
    assert pres[1].shape == posts[1].shape == (10,)


def test_forward_batch_shape() -> None:
    rng = np.random.default_rng(0)
    mlp = init_mlp([30, 20, 10], [Activation.TANH, Activation.LINEAR], 0.01, rng)
    X = rng.normal(size=(50, 30))
    Y = mlp.forward_batch(X)
    assert Y.shape == (50, 10)


def test_step_signal_shapes_fa() -> None:
    rng = np.random.default_rng(0)
    mlp = init_mlp([30, 20, 10], [Activation.TANH, Activation.LINEAR], 0.01, rng, use_bias=True)
    B = uniform((20, 10), 0.5, rng)
    x = rng.normal(size=30)
    y_star = rng.normal(size=10)
    result = step(mlp, x, y_star, alg="fa", eta=1e-3, feedback_matrices=[B])
    assert result.y.shape == (10,)
    assert result.signals_fa is not None
    assert result.signals_bp[0].shape == (20,) and result.signals_bp[1].shape == (10,)
    assert result.signals_fa[0].shape == (20,) and result.signals_fa[1].shape == (10,)


def test_step_signal_shapes_deep_fa() -> None:
    """Deep cascade: feedback matrices align with layer.n_out pairs."""
    rng = np.random.default_rng(0)
    sizes = [30, 20, 10, 10]
    acts = [Activation.TANH, Activation.TANH, Activation.LINEAR]
    mlp = init_mlp(sizes, acts, 0.01, rng, use_bias=True)
    feedback = [uniform((20, 10), 0.1, rng), uniform((10, 10), 0.1, rng)]
    x = rng.normal(size=30)
    y_star = rng.normal(size=10)
    result = step(mlp, x, y_star, alg="fa", eta=1e-3, feedback_matrices=feedback)
    assert result.signals_fa is not None
    assert [s.shape for s in result.signals_fa] == [(20,), (10,), (10,)]
    assert [s.shape for s in result.signals_bp] == [(20,), (10,), (10,)]


def test_step_feedback_length_must_match_depth_minus_one() -> None:
    rng = np.random.default_rng(0)
    mlp = init_mlp([30, 20, 10], [Activation.TANH, Activation.LINEAR], 0.01, rng)
    bad_feedback = [uniform((20, 10), 0.5, rng), uniform((10, 10), 0.5, rng)]  # wrong length
    x = rng.normal(size=30)
    y_star = rng.normal(size=10)
    try:
        step(mlp, x, y_star, alg="fa", eta=1e-3, feedback_matrices=bad_feedback)
    except ValueError as e:
        assert "feedback_matrices" in str(e)
    else:
        raise AssertionError("expected ValueError for mismatched feedback length")


def test_step_fa_without_feedback_raises() -> None:
    rng = np.random.default_rng(0)
    mlp = init_mlp([30, 20, 10], [Activation.TANH, Activation.LINEAR], 0.01, rng)
    x = rng.normal(size=30)
    y_star = rng.normal(size=10)
    try:
        step(mlp, x, y_star, alg="fa", eta=1e-3, feedback_matrices=None)
    except ValueError as e:
        assert "feedback_matrices" in str(e)
    else:
        raise AssertionError("expected ValueError when FA is missing feedback_matrices")
