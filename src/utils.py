import math
import numpy as np

def set_seed(seed: int) -> None:
    np.random.seed(seed)

def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out

def sigmoid_prime_from_act(a: np.ndarray) -> np.ndarray:
    # derivative given activation a = sigmoid(z)
    return a * (1.0 - a)

def tanh_prime_from_act(a: np.ndarray) -> np.ndarray:
    # derivative given activation a = tanh(z)
    return 1.0 - a**2

def angle(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    θ = arccos(|a^T b| / (||a||·||b||))
    Matches paper definition with scalar inner-product magnitude.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    denom = max(eps, na * nb)
    cosv = abs(float(np.dot(a, b))) / denom
    cosv = min(1.0, max(0.0, cosv))
    return float(math.acos(cosv))

def nse(y_star: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Normalized squared error used in paper plots:
        NSE = ||y* - y||^2 / ||y*||^2
    This yields ~1 at initialization when y≈0 and y* dominates.
    """
    y_star = np.asarray(y_star, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    num = float(np.dot(y_star - y, y_star - y))
    den = float(np.dot(y_star, y_star))
    return num / max(eps, den)

def one_hot(label: int, n_classes: int = 10) -> np.ndarray:
    v = np.zeros((n_classes,), dtype=np.float64)
    v[int(label)] = 1.0
    return v

def uniform_matrix(shape, scale, rng: np.random.Generator):
    return rng.uniform(low=-scale, high=scale, size=shape).astype(np.float64)

def mask50(shape, rng: np.random.Generator):
    # 50% entries kept (True), 50% removed (False)
    return (rng.random(shape) < 0.5).astype(np.float64)
