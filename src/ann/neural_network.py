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
    """

    def __init__(self, cli_args):
        """
        Initialize the neural network from CLI arguments.

        Args:
            cli_args: argparse Namespace with all config parameters
        """
        self.cli_args = cli_args

        # resolve hidden sizes
        if len(cli_args.hidden_size) == 1:
            self.hidden_sizes = [cli_args.hidden_size[0]] * cli_args.num_layers
        elif len(cli_args.hidden_size) == cli_args.num_layers:
            self.hidden_sizes = cli_args.hidden_size
        else:
            raise ValueError(
                f"--hidden_size must have 1 value or exactly "
                f"{cli_args.num_layers} values."
            )

        self.input_size  = 784
        self.num_classes = 10
        self.activation  = cli_args.activation
        self.weight_init = cli_args.weight_init
        self.loss_name   = cli_args.loss

        self.loss_fn, self.loss_grad = get_loss(cli_args.loss)
        self.layers = self._build_layers()

        # gradient storage — exposed for autograder
        self.grad_W = None
        self.grad_b = None

    # ── architecture ──────────────────────────────────────────────────────────
    def _build_layers(self):
        layers = []
        sizes  = [self.input_size] + self.hidden_sizes + [self.num_classes]

        for i in range(len(sizes) - 1):
            is_output = (i == len(sizes) - 2)
            # output layer has NO activation — returns logits
            act = "none" if is_output else self.activation
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
        Returns logits (linear combination, no softmax applied).
        X shape: (batch_size, input_size)
        Output shape: (batch_size, num_classes)
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A   # raw logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns softmax probabilities for evaluation."""
        from .activations import softmax
        logits = self.forward(X)
        return softmax(logits)

    # ── backward pass ─────────────────────────────────────────────────────────
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray,
             weight_decay: float = 0.0):
        from .activations import softmax

        # apply softmax to logits to get probs
        probs = softmax(y_pred)

        # combined softmax + CE gradient = (probs - y_true) / batch_size
        # pass this directly as dZ to output layer (bypassing activation grad)
        batch_size = y_true.shape[0]
        dZ = (probs - y_true)

        grad_W_list = []
        grad_b_list = []

        ## output layer gradients
        output_layer = self.layers[-1]
        output_layer.grad_W = (output_layer.A_prev.T @ dZ) / batch_size
        output_layer.grad_b = np.mean(dZ, axis=0, keepdims=True)
        grad_W_list.append(output_layer.grad_W)
        grad_b_list.append(output_layer.grad_b)

        # propagate to previous layers — dA is unnormalised
        dA = dZ @ output_layer.W.T

        for layer in reversed(self.layers[:-1]):
            dA = layer.backward(dA, weight_decay=weight_decay)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # store as object arrays — index 0 = last layer
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    # ── weight update ──────────────────────────────────────────────────────────
    def update_weights(self, optimizer):
        """Update weights using the optimizer."""
        if hasattr(optimizer, "step_start"):
            optimizer.step_start()
        for layer in self.layers:
            optimizer.update(layer)

    # ── loss ──────────────────────────────────────────────────────────────────
    def compute_loss(self, logits: np.ndarray, y_true: np.ndarray,
                     weight_decay: float = 0.0) -> float:
        """Compute loss from logits (applies softmax internally)."""
        from .activations import softmax
        probs     = softmax(logits)
        data_loss = self.loss_fn(probs, y_true)
        if weight_decay > 0:
            l2 = sum(np.sum(layer.W ** 2) for layer in self.layers)
            data_loss += 0.5 * weight_decay * l2
        return data_loss

    # ── train ─────────────────────────────────────────────────────────────────
    def train(self, X_train, y_train, epochs=1, batch_size=32,
              optimizer=None, X_val=None, y_val=None,
              weight_decay=0.0, wandb_log=False):
        """Train the network."""
        from utils.data_loader import get_batches

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches  = 0

            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                logits = self.forward(X_batch)
                epoch_loss += self.compute_loss(logits, y_batch, weight_decay)
                self.backward(y_batch, logits, weight_decay)
                self.update_weights(optimizer)
                n_batches += 1

            train_loss = epoch_loss / n_batches
            train_acc  = self.evaluate(X_train, y_train)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            log_dict = {
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc
            }

            if X_val is not None:
                val_logits = self.forward(X_val)
                val_loss   = self.compute_loss(val_logits, y_val, weight_decay)
                val_acc    = self.evaluate(X_val, y_val)
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
        """Returns accuracy."""
        probs  = self.predict_proba(X)
        preds  = np.argmax(probs, axis=1)
        labels = np.argmax(y,     axis=1)
        return float(np.mean(preds == labels))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ── get / set weights ─────────────────────────────────────────────────────
    def get_weights(self):
        """Return all weights as a dict."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict: dict):
        """Load weights from a dict."""
        for i, layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.W = weight_dict[f"W{i}"].copy()
            if f"b{i}" in weight_dict:
                layer.b = weight_dict[f"b{i}"].copy()

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path: str):
        """Save weights to .npy file."""
        np.save(path, self.get_weights())
        print(f"[Model] Saved → {path}")

    def load(self, path: str):
        """Load weights from .npy file."""
        data = np.load(path, allow_pickle=True).item()
        self.set_weights(data)
        print(f"[Model] Loaded ← {path}")

    # ── numerical gradient check ──────────────────────────────────────────────
    def numerical_gradients(self, X, y_true, weight_decay=0.0, eps=1e-5):
        """Central difference numerical gradients for gradient checking."""
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