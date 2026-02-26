"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import NeuralLayer
from .objective_functions import get_loss


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.

    Parameters
    ----------
    input_size   : int       – number of input features (784 for MNIST)
    hidden_sizes : list[int] – neurons per hidden layer e.g. [128, 128, 128]
    num_classes  : int       – output neurons (10 for MNIST/Fashion-MNIST)
    activation   : str       – hidden layer activation: 'sigmoid'|'tanh'|'relu'
    weight_init  : str       – 'random' | 'xavier'
    loss         : str       – 'cross_entropy' | 'mse'
    """

    def __init__(self, input_size: int, hidden_sizes: list,
                 num_classes: int = 10, activation: str = "relu",
                 weight_init: str = "xavier", loss: str = "cross_entropy"):

        self.input_size   = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes  = num_classes
        self.activation   = activation
        self.weight_init  = weight_init
        self.loss_name    = loss

        self.loss_fn, self.loss_grad = get_loss(loss)
        self.layers = self._build_layers()

    # ── architecture ──────────────────────────────────────────────────────────
    def _build_layers(self):
        layers = []
        sizes  = [self.input_size] + self.hidden_sizes + [self.num_classes]

        for i in range(len(sizes) - 1):
            is_output = (i == len(sizes) - 2)
            act = "softmax" if is_output else self.activation
            layers.append(
                NeuralLayer(
                    in_features  = sizes[i],
                    out_features = sizes[i + 1],
                    activation   = act,
                    weight_init  = self.weight_init,
                )
            )
        return layers

    # ── forward pass ──────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through all layers.

        Args:
            X : (batch_size, input_size)

        Returns:
            softmax probabilities : (batch_size, num_classes)
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    # ── backward pass ─────────────────────────────────────────────────────────
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray,
                 weight_decay: float = 0.0):
        """
        Backward propagation to compute and store gradients in every layer.

        Args:
            y_true       : (batch, num_classes) – one-hot targets
            y_pred       : (batch, num_classes) – network output (softmax probs)
            weight_decay : L2 regularisation coefficient

        Returns:
            grad_W, grad_b of the FIRST layer (for compatibility)
        """
        # upstream gradient: dL/d(probs)
        dA = self.loss_grad(y_pred, y_true)

        for layer in reversed(self.layers):
            dA = layer.backward(dA, weight_decay=weight_decay)

        return self.layers[0].grad_W, self.layers[0].grad_b

    # ── weight update ──────────────────────────────────────────────────────────
    def update_weights(self, optimizer):
        """
        Update weights using the optimizer.

        Args:
            optimizer : any optimizer instance from optimizers.py
        """
        if hasattr(optimizer, "step_start"):
            optimizer.step_start()

        for layer in self.layers:
            optimizer.update(layer)

    # ── loss ──────────────────────────────────────────────────────────────────
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray,
                     weight_decay: float = 0.0) -> float:
        """Returns scalar loss + optional L2 regularisation term."""
        data_loss = self.loss_fn(y_pred, y_true)
        if weight_decay > 0:
            l2 = sum(np.sum(layer.W ** 2) for layer in self.layers)
            data_loss += 0.5 * weight_decay * l2
        return data_loss

    # ── train ─────────────────────────────────────────────────────────────────
    def train(self, X_train, y_train, epochs, batch_size,
              optimizer, X_val=None, y_val=None,
              weight_decay=0.0, wandb_log=False):
        """
        Train the network for specified epochs.

        Returns:
            history dict with train_loss, train_acc, val_loss, val_acc per epoch
        """
        from src.utils.data_loader import get_batches

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches  = 0

            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                y_pred = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_pred, y_batch, weight_decay)
                self.backward(y_batch, y_pred, weight_decay)
                self.update_weights(optimizer)
                n_batches += 1

            # ── epoch metrics ─────────────────────────────────────────────────
            train_loss = epoch_loss / n_batches
            train_acc  = self.evaluate(X_train, y_train)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            log_dict = {
                "epoch":      epoch,
                "train_loss": train_loss,
                "train_acc":  train_acc,
            }

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(val_pred, y_val, weight_decay)
                val_acc  = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                log_dict["val_loss"] = val_loss
                log_dict["val_acc"]  = val_acc

                print(
                    f"Epoch [{epoch:>3}/{epochs}]  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch:>3}/{epochs}]  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                )

            if wandb_log:
                try:
                    import wandb
                    wandb.log(log_dict)
                except ImportError:
                    pass

        return history

    # ── evaluate ──────────────────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate accuracy on given data.

        Args:
            X : input features
            y : one-hot labels

        Returns:
            accuracy as a float between 0 and 1
        """
        probs  = self.forward(X)
        preds  = np.argmax(probs,  axis=1)
        labels = np.argmax(y,      axis=1)
        return float(np.mean(preds == labels))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return np.argmax(self.forward(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities."""
        return self.forward(X)

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path: str):
        """Save all layer weights to a single .npy file."""
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"layer_{i}_W"] = layer.W
            params[f"layer_{i}_b"] = layer.b
        np.save(path, params)
        print(f"[Model] Saved → {path}")

    def load(self, path: str):
        """Load weights from a .npy file produced by save()."""
        params = np.load(path, allow_pickle=True).item()
        for i, layer in enumerate(self.layers):
            layer.W = params[f"layer_{i}_W"]
            layer.b = params[f"layer_{i}_b"]
        print(f"[Model] Loaded ← {path}")

    # ── gradient check (autograder Q1.2) ─────────────────────────────────────
    def numerical_gradients(self, X: np.ndarray, y_true: np.ndarray,
                             weight_decay: float = 0.0, eps: float = 1e-5):
        """
        Central-difference numerical gradients for verifying backward().
        Returns list of dicts [{"W": ..., "b": ...}] for each layer.
        Tolerance required by autograder: 1e-7
        """
        num_grads = []

        for layer in self.layers:
            dW = np.zeros_like(layer.W)
            db = np.zeros_like(layer.b)

            it = np.nditer(layer.W, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = layer.W[idx]
                layer.W[idx] = orig + eps
                lp = self.compute_loss(self.forward(X), y_true, weight_decay)
                layer.W[idx] = orig - eps
                lm = self.compute_loss(self.forward(X), y_true, weight_decay)
                dW[idx] = (lp - lm) / (2 * eps)
                layer.W[idx] = orig
                it.iternext()

            it = np.nditer(layer.b, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = layer.b[idx]
                layer.b[idx] = orig + eps
                lp = self.compute_loss(self.forward(X), y_true, weight_decay)
                layer.b[idx] = orig - eps
                lm = self.compute_loss(self.forward(X), y_true, weight_decay)
                db[idx] = (lp - lm) / (2 * eps)
                layer.b[idx] = orig
                it.iternext()

            num_grads.append({"W": dW, "b": db})

        return num_grads