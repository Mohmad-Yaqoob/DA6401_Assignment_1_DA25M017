"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from .activations import get_activation, softmax


class NeuralLayer:
    """
    One fully-connected (dense) layer.

    Parameters
    ----------
    in_features  : int  – number of input neurons
    out_features : int  – number of output neurons
    activation   : str  – 'sigmoid' | 'tanh' | 'relu' | 'softmax'
    weight_init  : str  – 'random' | 'xavier'
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = "relu", weight_init: str = "xavier"):

        self.in_features  = in_features
        self.out_features = out_features
        self.activation   = activation.lower()
        self.weight_init  = weight_init.lower()

        # initialise weights & biases
        self.W, self.b = self._init_weights()

        # cache filled during forward pass – needed for backward
        self.Z      = None   # pre-activation  (batch, out_features)
        self.A      = None   # post-activation (batch, out_features)
        self.A_prev = None   # input to this layer

        # gradients – exposed so autograder can inspect self.grad_W / self.grad_b
        self.grad_W = None
        self.grad_b = None

    # ── weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        if self.weight_init == "xavier":
            limit = np.sqrt(6.0 / (self.in_features + self.out_features))
            W = np.random.uniform(-limit, limit,
                                  (self.in_features, self.out_features))
        elif self.weight_init == "random":
            W = np.random.randn(self.in_features, self.out_features) * 0.01
        elif self.weight_init == "zeros":
            # used only for the symmetry-breaking experiment
            W = np.zeros((self.in_features, self.out_features))
        else:
            raise ValueError(f"Unknown weight_init '{self.weight_init}'.")

        b = np.zeros((1, self.out_features))
        return W, b

    # ── forward pass ──────────────────────────────────────────────────────────
    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """
        A_prev : (batch_size, in_features)
        Returns A : (batch_size, out_features)
        """
        self.A_prev = A_prev
        self.Z      = A_prev @ self.W + self.b

        if self.activation == "softmax":
            self.A = softmax(self.Z)
        else:
            act_fn, _ = get_activation(self.activation)
            self.A = act_fn(self.Z)

        return self.A

    # ── backward pass ─────────────────────────────────────────────────────────
    def backward(self, dA: np.ndarray, weight_decay: float = 0.0) -> np.ndarray:
        """
        Backprop through this layer.

        dA           : upstream gradient w.r.t this layer's output A
                       shape (batch_size, out_features)
        weight_decay : L2 regularisation coefficient λ

        Returns dA_prev : gradient w.r.t A_prev (batch_size, in_features)
        Sets    self.grad_W, self.grad_b
        """
        batch_size = self.A_prev.shape[0]

        # ── gradient of loss w.r.t pre-activation Z ──────────────────────────
        if self.activation == "softmax":
            # dA here is dL/d(probs).
            # Apply softmax Jacobian: dZ_i = p_i * (dA_i - sum_j(dA_j * p_j))
            probs = self.A
            dZ = probs * (dA - np.sum(dA * probs, axis=1, keepdims=True))
        else:
            _, grad_fn = get_activation(self.activation)
            dZ = dA * grad_fn(self.Z)   # element-wise chain rule

        # ── gradients w.r.t parameters ───────────────────────────────────────
        self.grad_W = (self.A_prev.T @ dZ) / batch_size
        self.grad_b = np.mean(dZ, axis=0, keepdims=True)

        # L2 regularisation term on weights (never on biases)
        if weight_decay > 0:
            self.grad_W += weight_decay * self.W

        # ── gradient to propagate backwards ──────────────────────────────────
        dA_prev = dZ @ self.W.T
        return dA_prev

    # ── serialisation helpers ─────────────────────────────────────────────────
    def get_params(self):
        return {"W": self.W, "b": self.b}

    def set_params(self, params: dict):
        self.W = params["W"]
        self.b = params["b"]