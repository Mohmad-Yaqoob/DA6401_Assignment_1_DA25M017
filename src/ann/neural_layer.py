import numpy as np
from .activations import get_activation


class NeuralLayer:

    def __init__(self, in_features, out_features,
                 activation="relu", weight_init="xavier"):

        self.in_features  = in_features
        self.out_features = out_features
        self.weight_init  = weight_init

        # Works whether activation is a string ("relu") or an object (ReLU())
        self.activation = get_activation(activation)

        self.W, self.b = self._init_weights()

        self.Z      = None
        self.x      = None   # alias used by NeuralNetwork
        self.A_prev = None   # alias used by NeuralNetwork

        self.grad_W = np.zeros((in_features, out_features))
        self.grad_b = np.zeros((1, out_features))
        self.optimizer_state = {}


    def _init_weights(self):
        if self.weight_init == "xavier":
            limit = np.sqrt(6.0 / (self.in_features + self.out_features))
            W = np.random.uniform(-limit, limit, (self.in_features, self.out_features))
        elif self.weight_init == "random":
            W = np.random.randn(self.in_features, self.out_features) * 0.01
        else:
            W = np.zeros((self.in_features, self.out_features))

        b = np.zeros((1, self.out_features))
        return W, b


    def forward(self, A_prev):
        self.A_prev = A_prev
        self.x      = A_prev          # keep both aliases in sync
        self.Z      = A_prev @ self.W + self.b
        self.z      = self.Z          # alias
        return self.activation.forward(self.Z)


    def backward(self, grad_output, weight_decay=0.0):
        # activation.backward already multiplies grad_output * local_grad
        grad_z = self.activation.backward(grad_output)

        self.grad_W = self.A_prev.T @ grad_z
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True)

        if weight_decay > 0:
            self.grad_W += weight_decay * self.W

        return grad_z @ self.W.T