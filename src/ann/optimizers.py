"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
"""

import numpy as np


class SGD:
    """Vanilla stochastic gradient descent."""

    def __init__(self, lr=0.01, weight_decay=0.0, **kwargs):
        self.lr           = lr
        self.weight_decay = weight_decay

    def update(self, layer):
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b


class Momentum:
    """SGD with momentum (heavy ball method)."""

    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0, **kwargs):
        self.lr           = lr
        self.beta         = beta
        self.weight_decay = weight_decay
        self._state       = {}

    def _get_state(self, layer):
        lid = id(layer)
        if lid not in self._state:
            self._state[lid] = {
                "vW": np.zeros_like(layer.W),
                "vb": np.zeros_like(layer.b),
            }
        return self._state[lid]

    def update(self, layer):
        s = self._get_state(layer)
        s["vW"] = self.beta * s["vW"] + self.lr * layer.grad_W
        s["vb"] = self.beta * s["vb"] + self.lr * layer.grad_b
        layer.W -= s["vW"]
        layer.b -= s["vb"]


class NAG:
    """
    Nesterov Accelerated Gradient.
    Applies the Nesterov correction:
      v_new = beta*v_old + lr*grad
      W     = W - (-beta*v_old + (1+beta)*v_new)
    """

    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0, **kwargs):
        self.lr           = lr
        self.beta         = beta
        self.weight_decay = weight_decay
        self._state       = {}

    def _get_state(self, layer):
        lid = id(layer)
        if lid not in self._state:
            self._state[lid] = {
                "vW": np.zeros_like(layer.W),
                "vb": np.zeros_like(layer.b),
            }
        return self._state[lid]

    def update(self, layer):
        s = self._get_state(layer)

        vW_prev = s["vW"].copy()
        vb_prev = s["vb"].copy()

        s["vW"] = self.beta * s["vW"] + self.lr * layer.grad_W
        s["vb"] = self.beta * s["vb"] + self.lr * layer.grad_b

        layer.W -= -self.beta * vW_prev + (1 + self.beta) * s["vW"]
        layer.b -= -self.beta * vb_prev + (1 + self.beta) * s["vb"]


class RMSProp:
    """RMSProp: adapts learning rate using running average of squared gradients."""

    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8,
                 weight_decay=0.0, **kwargs):
        self.lr           = lr
        self.beta         = beta
        self.epsilon      = epsilon
        self.weight_decay = weight_decay
        self._state       = {}

    def _get_state(self, layer):
        lid = id(layer)
        if lid not in self._state:
            self._state[lid] = {
                "sW": np.zeros_like(layer.W),
                "sb": np.zeros_like(layer.b),
            }
        return self._state[lid]

    def update(self, layer):
        s = self._get_state(layer)

        s["sW"] = self.beta * s["sW"] + (1 - self.beta) * layer.grad_W ** 2
        s["sb"] = self.beta * s["sb"] + (1 - self.beta) * layer.grad_b ** 2

        layer.W -= self.lr * layer.grad_W / (np.sqrt(s["sW"]) + self.epsilon)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(s["sb"]) + self.epsilon)


class Adam:
    """
    Adam: Adaptive Moment Estimation.
    Combines momentum (1st moment) + RMSProp (2nd moment) with bias correction.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.0, **kwargs):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.epsilon      = epsilon
        self.weight_decay = weight_decay
        self._state       = {}
        self._t           = 0   # global step counter

    def _get_state(self, layer):
        lid = id(layer)
        if lid not in self._state:
            self._state[lid] = {
                "mW": np.zeros_like(layer.W),
                "vW": np.zeros_like(layer.W),
                "mb": np.zeros_like(layer.b),
                "vb": np.zeros_like(layer.b),
            }
        return self._state[lid]

    def step_start(self):
        """Call once per batch BEFORE updating all layers."""
        self._t += 1

    def update(self, layer):
        s  = self._get_state(layer)
        t  = self._t
        b1 = self.beta1
        b2 = self.beta2

        # biased moment estimates
        s["mW"] = b1 * s["mW"] + (1 - b1) * layer.grad_W
        s["vW"] = b2 * s["vW"] + (1 - b2) * layer.grad_W ** 2
        s["mb"] = b1 * s["mb"] + (1 - b1) * layer.grad_b
        s["vb"] = b2 * s["vb"] + (1 - b2) * layer.grad_b ** 2

        # bias-corrected estimates
        mW_hat = s["mW"] / (1 - b1 ** t)
        vW_hat = s["vW"] / (1 - b2 ** t)
        mb_hat = s["mb"] / (1 - b1 ** t)
        vb_hat = s["vb"] / (1 - b2 ** t)

        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.epsilon)


class Nadam:
    """
    Nadam: Adam with Nesterov momentum.
    Replaces the first moment estimate with a Nesterov look-ahead version.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.0, **kwargs):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.epsilon      = epsilon
        self.weight_decay = weight_decay
        self._state       = {}
        self._t           = 0

    def _get_state(self, layer):
        lid = id(layer)
        if lid not in self._state:
            self._state[lid] = {
                "mW": np.zeros_like(layer.W),
                "vW": np.zeros_like(layer.W),
                "mb": np.zeros_like(layer.b),
                "vb": np.zeros_like(layer.b),
            }
        return self._state[lid]

    def step_start(self):
        """Call once per batch BEFORE updating all layers."""
        self._t += 1

    def update(self, layer):
        s  = self._get_state(layer)
        t  = self._t
        b1 = self.beta1
        b2 = self.beta2

        s["mW"] = b1 * s["mW"] + (1 - b1) * layer.grad_W
        s["vW"] = b2 * s["vW"] + (1 - b2) * layer.grad_W ** 2
        s["mb"] = b1 * s["mb"] + (1 - b1) * layer.grad_b
        s["vb"] = b2 * s["vb"] + (1 - b2) * layer.grad_b ** 2

        mW_hat = s["mW"] / (1 - b1 ** t)
        vW_hat = s["vW"] / (1 - b2 ** t)
        mb_hat = s["mb"] / (1 - b1 ** t)
        vb_hat = s["vb"] / (1 - b2 ** t)

        # Nesterov correction: look-ahead first moment
        mW_nadam = b1 * mW_hat + (1 - b1) * layer.grad_W / (1 - b1 ** t)
        mb_nadam = b1 * mb_hat + (1 - b1) * layer.grad_b / (1 - b1 ** t)

        layer.W -= self.lr * mW_nadam / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.lr * mb_nadam / (np.sqrt(vb_hat) + self.epsilon)


# ------------------------------------------------------------------
# Lookup so the rest of the code can reference optimizers by name
# ------------------------------------------------------------------

OPTIMIZER_MAP = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
    "adam":     Adam,
    "nadam":    Nadam,
}


def get_optimizer(name: str, **kwargs):
    """Instantiate and return an optimizer by name."""
    name = name.lower()
    if name not in OPTIMIZER_MAP:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: {list(OPTIMIZER_MAP.keys())}"
        )
    return OPTIMIZER_MAP[name](**kwargs)