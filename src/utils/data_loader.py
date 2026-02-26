"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
def load_dataset(name: str, val_split: float = 0.1, seed: int = 42):
    """
    Load MNIST or Fashion-MNIST, normalise, one-hot encode, and split.

    Parameters
    ----------
    name      : 'mnist' or 'fashion_mnist'
    val_split : fraction of training data used for validation
    seed      : random seed for reproducibility

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    - X arrays : (N, 784)  float32  normalised to [0, 1]
    - y arrays : (N, 10)   float32  one-hot encoded
    """
    name = name.lower().replace("-", "_")

    if name == "mnist":
        (X_tr, y_tr), (X_te, y_te) = mnist.load_data()
    elif name == "fashion_mnist":
        (X_tr, y_tr), (X_te, y_te) = fashion_mnist.load_data()
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: 'mnist' or 'fashion_mnist'."
        )

    # flatten 28x28 → 784 and normalise to [0, 1]
    X_tr = X_tr.reshape(X_tr.shape[0], -1).astype(np.float32) / 255.0
    X_te = X_te.reshape(X_te.shape[0], -1).astype(np.float32) / 255.0

    # stratified train / val split — preserves class balance
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr,
        test_size  = val_split,
        random_state = seed,
        stratify   = y_tr,
    )

    # one-hot encode
    y_train = one_hot(y_tr,  num_classes=10)
    y_val   = one_hot(y_val, num_classes=10)
    y_test  = one_hot(y_te,  num_classes=10)

    return X_tr, y_train, X_val, y_val, X_te, y_test


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert integer class labels to one-hot matrix.

    Parameters
    ----------
    labels      : (N,) integer array
    num_classes : number of classes

    Returns
    -------
    (N, num_classes) float32 array
    """
    n   = labels.shape[0]
    out = np.zeros((n, num_classes), dtype=np.float32)
    out[np.arange(n), labels] = 1.0
    return out


def get_batches(X: np.ndarray, y: np.ndarray,
                batch_size: int, shuffle: bool = True):
    """
    Generator that yields (X_batch, y_batch) mini-batches.
    Shuffles indices each call when shuffle=True.
    Last incomplete batch is dropped to keep batch sizes consistent.

    Parameters
    ----------
    X          : (N, features)
    y          : (N, classes)
    batch_size : int
    shuffle    : bool
    """
    n       = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n - batch_size + 1, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]