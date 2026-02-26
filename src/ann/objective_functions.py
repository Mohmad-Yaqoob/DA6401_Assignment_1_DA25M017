"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""


"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

EPS = 1e-12  # prevents log(0)


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Categorical cross-entropy loss.
    y_pred : (batch, classes) – softmax probabilities
    y_true : (batch, classes) – one-hot labels
    """
    batch_size = y_pred.shape[0]
    log_probs  = -np.sum(y_true * np.log(y_pred + EPS), axis=1)
    return float(np.mean(log_probs))


def cross_entropy_backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Gradient of cross-entropy w.r.t softmax probabilities: dL/d(probs).
    = -y_true / probs  (unnormalised)
    NeuralLayer.backward() applies the softmax Jacobian and divides by batch.
    """
    return -y_true / (y_pred + EPS)


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean squared error averaged over batch and classes.
    y_pred : (batch, classes) – softmax probabilities
    y_true : (batch, classes) – one-hot labels
    """
    return float(np.mean((y_pred - y_true) ** 2))


def mse_backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE w.r.t y_pred (post-softmax).
    Returns unnormalised gradient — NeuralLayer.backward() handles /batch_size.
    """
    num_classes = y_pred.shape[1]
    return 2.0 * (y_pred - y_true) / num_classes


# ------------------------------------------------------------------
# Lookup so the rest of the code can reference losses by name
# ------------------------------------------------------------------

LOSS_MAP = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_backward),
    "mse":           (mse_loss,           mse_backward),
}


def get_loss(name: str):
    """Return (loss_fn, grad_fn) for the given loss name."""
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name not in LOSS_MAP:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSS_MAP.keys())}")
    return LOSS_MAP[name]