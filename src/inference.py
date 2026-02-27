"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
import os
import sys
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann import NeuralNetwork
from utils import load_dataset


# ── argument parser ────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    # W&B
    parser.add_argument("-wp", "--wandb_project",
                        type=str, default="da6401_assignment_1")
    parser.add_argument("-we", "--wandb_entity",
                        type=str, default=None)

    # dataset
    parser.add_argument("-d",  "--dataset",
                        type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    # architecture — same defaults as best model
    parser.add_argument("-nhl", "--num_layers",
                        type=int, default=3)
    parser.add_argument("-sz",  "--hidden_size",
                        type=int, nargs="+", default=[128])
    parser.add_argument("-a",   "--activation",
                        type=str, default="relu",
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-wi",  "--weight_init",
                        type=str, default="xavier",
                        choices=["random", "xavier"])
    parser.add_argument("-l",   "--loss",
                        type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-o",   "--optimizer",
                        type=str, default="adam",
                        choices=["sgd", "momentum", "nag",
                                 "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr",  "--learning_rate",
                        type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",
                        type=float, default=0.0)
    parser.add_argument("-e",   "--epochs",
                        type=int, default=15)
    parser.add_argument("-b",   "--batch_size",
                        type=int, default=64)

    # paths
    parser.add_argument("--model_path",
                        type=str, default="models/best_model.npy",
                        help="Relative path to saved model weights")
    parser.add_argument("--config_path",
                        type=str, default="models/best_config.json")
    parser.add_argument("--no_wandb",
                        action="store_true")

    return parser.parse_args()


# ── load model ─────────────────────────────────────────────────────────────────

def load_model(model_path):
    """Load trained model weights from disk."""
    data = np.load(model_path, allow_pickle=True).item()
    return data


# ── evaluate model ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.

    Returns
    -------
    dict with keys: logits, loss, accuracy, f1, precision, recall
    """
    logits     = model.forward(X_test)
    probs      = model.predict_proba(X_test)
    y_pred_int = np.argmax(probs,  axis=1)
    y_true_int = np.argmax(y_test, axis=1)

    loss      = model.compute_loss(logits, y_test)
    accuracy  = accuracy_score (y_true_int, y_pred_int)
    precision = precision_score(y_true_int, y_pred_int,
                                average="macro", zero_division=0)
    recall    = recall_score   (y_true_int, y_pred_int,
                                average="macro", zero_division=0)
    f1        = f1_score       (y_true_int, y_pred_int,
                                average="macro", zero_division=0)

    return {
        "logits":    logits,
        "loss":      loss,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    """
    Main inference function.
    Returns dict with logits, loss, accuracy, f1, precision, recall.
    """
    args = parse_arguments()

    # override args from config file if it exists
    if os.path.exists(args.config_path):
        with open(args.config_path) as f:
            cfg = json.load(f)
        args.num_layers   = cfg.get("num_layers",    args.num_layers)
        args.hidden_size  = cfg.get("hidden_size",   args.hidden_size)
        args.activation   = cfg.get("activation",    args.activation)
        args.weight_init  = cfg.get("weight_init",   args.weight_init)
        args.loss         = cfg.get("loss",          args.loss)
        args.dataset      = cfg.get("dataset",       args.dataset)

    # ── load model ────────────────────────────────────────────────────────────
    model   = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    # ── load data ─────────────────────────────────────────────────────────────
    _, _, _, _, X_test, y_test = load_dataset(args.dataset)

    # ── evaluate ──────────────────────────────────────────────────────────────
    results = evaluate_model(model, X_test, y_test)

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS — {args.dataset.upper()}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"{'='*50}\n")

    # ── W&B logging ───────────────────────────────────────────────────────────
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project  = args.wandb_project,
                entity   = args.wandb_entity,
                job_type = "inference",
            )
            wandb.log({
                "test_accuracy":  results["accuracy"],
                "test_precision": results["precision"],
                "test_recall":    results["recall"],
                "test_f1":        results["f1"],
                "test_loss":      results["loss"],
            })
            wandb.finish()
        except ImportError:
            pass

    print("Evaluation complete!")
    return results


if __name__ == "__main__":
    main()