"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


def sigmoid(Z):
    # overflow control
    Z = np.clip(Z, -500, 500)
    # 1/(1 + e^-z)
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_backward(Z):
    # derivattive of sigmoid
    s = sigmoid(Z)
    return s * (1.0 - s)

def tanh(Z):
    # i used direct numpy tanh function
    return np.tanh(Z)

def tanh_backward(Z):
    # derivative of the tanH function
    return 1.0 - np.tanh(Z) ** 2

def relu(Z):
    # relu max(0, z)
    return np.maximum(0.0, Z)

def relu_backward(Z):
    # derivative of the Relu function
    return (Z > 0).astype(float)

def softmax(Z):
    # softmax along axis = 1
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

ACTIVATION_MAP = {
    "sigmoid": (sigmoid, sigmoid_backward),
    "tanh":    (tanh,    tanh_backward),
    "relu":    (relu,    relu_backward),
}

def get_activation(name: str):
    """Return (forward_fn, grad_fn) for the given activation name."""
    # helper function for returning the fwd and bwd of given activation name
    name = name.lower()
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(ACTIVATION_MAP.keys())}")
    return ACTIVATION_MAP[name]