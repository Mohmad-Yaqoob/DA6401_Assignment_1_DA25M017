import numpy as np
from .neural_layer import NeuralLayer
from .objective_functions import get_loss, get_loss_obj
from .activations import softmax


class NeuralNetwork:

    def __init__(self,
                 cli_args=None,
                 input_size=784,
                 hidden_sizes=None,
                 output_size=10,
                 activation="relu",
                 weight_init="xavier",
                 loss="cross_entropy",
                 num_layers=None,
                 hidden_size=None):

        import argparse

        if isinstance(cli_args, argparse.Namespace):
            activation  = cli_args.activation
            weight_init = cli_args.weight_init
            loss        = cli_args.loss
            num_layers  = getattr(cli_args, "num_layers", None)
            hidden_sizes = getattr(cli_args, "hidden_sizes",
                           getattr(cli_args, "hidden_size", None))

        # also handle if input_size itself is a Namespace (friend's style)
        if isinstance(input_size, argparse.Namespace):
            ns = input_size
            input_size   = getattr(ns, 'input_size', 784)
            hidden_sizes = getattr(ns, 'hidden_sizes', getattr(ns, 'hidden_size', None))
            output_size  = getattr(ns, 'output_size', 10)
            activation   = getattr(ns, 'activation', 'relu')
            weight_init  = getattr(ns, 'weight_init', 'xavier')
            loss         = getattr(ns, 'loss', 'cross_entropy')
            num_layers   = getattr(ns, 'num_layers', None)

        # ---- hidden layer processing ----
        if hidden_sizes is None and hidden_size is not None:
            hidden_sizes = hidden_size

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * (num_layers if num_layers else 1)
        elif isinstance(hidden_sizes, list):
            if num_layers is not None and len(hidden_sizes) == 1:
                hidden_sizes = hidden_sizes * num_layers
        
        if hidden_sizes is None:
            hidden_sizes = [128] * (int(num_layers) if num_layers else 1)

        if isinstance(hidden_sizes, (int, np.integer)):
            hidden_sizes = [int(hidden_sizes)]
        hidden_sizes = [int(h) for h in hidden_sizes]

        self.hidden_sizes = hidden_sizes
        self.input_size   = int(input_size)
        self.num_classes  = int(output_size)
        self.output_size  = int(output_size)
        self.activation   = activation
        self.weight_init  = weight_init
        self.loss_name    = str(loss)

        # support both tuple and object loss interfaces
        loss_result = get_loss(str(loss))
        if isinstance(loss_result, tuple):
            self.loss_fn, self.loss_grad = loss_result
        else:
            self.loss_fn   = loss_result.forward
            self.loss_grad = loss_result.backward

        # also keep object version for grader compatibility
        self.loss_obj = get_loss_obj(str(loss))

        self.layers = self._build_layers()

        self.grad_W = None
        self.grad_b = None

    def _build_layers(self):
        layers = []
        sizes = [self.input_size] + self.hidden_sizes + [self.num_classes]
        for i in range(len(sizes) - 1):
            is_output = (i == len(sizes) - 2)
            act = "none" if is_output else self.activation
            layers.append(NeuralLayer(
                in_features=sizes[i],
                out_features=sizes[i+1],
                activation=act,
                weight_init=self.weight_init
            ))
        return layers

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def predict_proba(self, X):
        logits = self.forward(X)
        return softmax(logits)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def compute_loss(self, logits, y_true, weight_decay=0.0):
        probs = softmax(logits)
        loss = self.loss_fn(probs, y_true)
        if weight_decay > 0:
            l2 = sum(np.sum(layer.W ** 2) for layer in self.layers)
            loss += 0.5 * weight_decay * l2
        return loss

    def backward(self, y_true, y_pred, weight_decay=0.0):
        probs = softmax(y_pred)
        batch_size = y_true.shape[0]

        y_true = np.array(y_true)
        if y_true.ndim == 1:
            dZ = probs.copy()
            dZ[np.arange(batch_size), y_true.astype(int)] -= 1
        else:
            dZ = probs - y_true
        dZ /= batch_size

        grad_W_list = []
        grad_b_list = []

        output_layer = self.layers[-1]
        output_layer.grad_W = output_layer.A_prev.T @ dZ
        output_layer.grad_b = np.sum(dZ, axis=0, keepdims=True)
        grad_W_list.append(output_layer.grad_W)
        grad_b_list.append(output_layer.grad_b)

        dA = dZ @ output_layer.W.T

        for layer in reversed(self.layers[:-1]):
            dA = layer.backward(dA, weight_decay=weight_decay)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        self.grad_W = grad_W_list[::-1]
        self.grad_b = grad_b_list[::-1]

        return self.grad_W, self.grad_b

    def update_weights(self, optimizer):
        if hasattr(optimizer, "step_start"):
            optimizer.step_start()
        for layer in self.layers:
            optimizer.update(layer)

    def train(self, X_train, y_train, epochs=1, batch_size=32,
              optimizer=None, X_val=None, y_val=None, weight_decay=0.0):

        from utils.data_loader import get_batches

        history = {"train_loss": [], "train_acc": []}

        for epoch in range(epochs):
            epoch_loss = 0
            n_batches  = 0

            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                logits = self.forward(X_batch)
                loss   = self.compute_loss(logits, y_batch, weight_decay)
                epoch_loss += loss
                self.backward(y_batch, logits, weight_decay)
                self.update_weights(optimizer)
                n_batches += 1

            train_loss = epoch_loss / n_batches
            train_acc  = self.evaluate(X_train, y_train)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            print(f"Epoch {epoch+1} | loss={train_loss:.4f} | acc={train_acc:.4f}")

        return history

    def evaluate(self, X, y):
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        labels = np.argmax(y, axis=1) if y.ndim == 2 else y
        return float(np.mean(preds == labels))

    def get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"W{i}"] = layer.W.copy()
            weights[f"b{i}"] = layer.b.copy()
        return weights

    def set_weights(self, weight_dict):
        if isinstance(weight_dict, np.ndarray) and weight_dict.ndim == 0:
            weight_dict = weight_dict.item()
        for i, layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.W = np.array(weight_dict[f"W{i}"]).copy()
            if f"b{i}" in weight_dict:
                layer.b = np.array(weight_dict[f"b{i}"]).copy()

    def save(self, path):
        import os, json
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        jw = {k: v.tolist() for k, v in self.get_weights().items()}
        json_path = path.replace('.npy', '.json')
        with open(json_path, 'w') as f:
            json.dump(jw, f)
        np.save(path, self.get_weights())
        print(f"[Model Saved] {path}")

    def load(self, path):
        import os, json
        json_path = path.replace('.npy', '.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                d = json.load(f)
            self.set_weights({k: np.array(v) for k, v in d.items()})
        else:
            data = np.load(path, allow_pickle=True)
            self.set_weights(data)
        print(f"[Model Loaded] {path}")