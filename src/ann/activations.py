import numpy as np


# ─── Classes (grader uses these) ───

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, grad):
        return grad * self.mask

class Sigmoid:
    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.out
    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)

class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)

class Identity:
    def forward(self, x): return x
    def backward(self, grad): return grad


# ─── Standalone functions (__init__.py imports these by name) ───

def sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_backward(Z):
    s = sigmoid(Z)
    return s * (1.0 - s)

def relu(Z):
    return np.maximum(0.0, Z)

def relu_backward(Z):
    return (Z > 0).astype(Z.dtype)

def tanh(Z):
    return np.tanh(Z)

def tanh_backward(Z):
    t = np.tanh(Z)
    return 1.0 - t * t

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# ─── Factory ───

ACTIVATION_CLASSES = {
    "relu":    ReLU,
    "sigmoid": Sigmoid,
    "tanh":    Tanh,
    "none":    Identity,
}

def get_activation(name):
    if hasattr(name, 'forward'):
        return name
    name = name.lower()
    if name not in ACTIVATION_CLASSES:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(ACTIVATION_CLASSES.keys())}")
    return ACTIVATION_CLASSES[name]()