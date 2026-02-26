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
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--model_path",
                        type=str, default="models/best_model.npy",
                        help="Relative path to saved model weights")
    parser.add_argument("--config_path",
                        type=str, default="models/best_config.json")
    parser.add_argument("-d", "--dataset",
                        type=str, default=None,
                        help="Override dataset from config (mnist | fashion_mnist)")
    parser.add_argument("--wandb_project",
                        type=str, default="da6401_assignment_1")
    parser.add_argument("--wandb_entity",
                        type=str, default=None)
    parser.add_argument("--no_wandb",
                        action="store_true")

    return parser.parse_args()


# ── load model ─────────────────────────────────────────────────────────────────

def load_model(model_path, config_path):
    """
    Load trained model from disk.
    Rebuilds architecture from config then loads weights.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    model = NeuralNetwork(
        input_size   = 784,
        hidden_sizes = cfg["hidden_sizes"],
        num_classes  = 10,
        activation   = cfg.get("activation",  "relu"),
        weight_init  = cfg.get("weight_init", "xavier"),
        loss         = cfg.get("loss",        "cross_entropy"),
    )
    model.load(model_path)
    return model, cfg


# ── evaluate model ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.

    Returns
    -------
    dict with keys: logits, loss, accuracy, f1, precision, recall
    """
    # forward pass
    probs      = model.predict_proba(X_test)
    y_pred_int = np.argmax(probs,   axis=1)
    y_true_int = np.argmax(y_test,  axis=1)

    # loss
    loss = model.compute_loss(probs, y_test)

    # metrics
    accuracy  = accuracy_score (y_true_int, y_pred_int)
    precision = precision_score(y_true_int, y_pred_int,
                                average="macro", zero_division=0)
    recall    = recall_score   (y_true_int, y_pred_int,
                                average="macro", zero_division=0)
    f1        = f1_score       (y_true_int, y_pred_int,
                                average="macro", zero_division=0)

    return {
        "logits":    probs,
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

    # ── load model ────────────────────────────────────────────────────────────
    model, cfg = load_model(args.model_path, args.config_path)

    # ── load data ─────────────────────────────────────────────────────────────
    dataset = args.dataset or cfg.get("dataset", "mnist")
    _, _, _, _, X_test, y_test = load_dataset(dataset)

    # ── evaluate ──────────────────────────────────────────────────────────────
    results = evaluate_model(model, X_test, y_test)

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS — {dataset.upper()}")
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