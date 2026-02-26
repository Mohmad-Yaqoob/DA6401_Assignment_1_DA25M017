"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann import NeuralNetwork, get_optimizer
from utils import load_dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed — logging to console only.")


# ── argument parser ────────────────────────────────────────────────────────────

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a neural network on MNIST / Fashion-MNIST")

    # W&B
    parser.add_argument("-wp", "--wandb_project",
                        type=str, default="da6401_assignment_1")
    parser.add_argument("-we", "--wandb_entity",
                        type=str, default=None)

    # dataset & training
    parser.add_argument("-d",   "--dataset",
                        type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",
                        type=int, default=10)
    parser.add_argument("-b",   "--batch_size",
                        type=int, default=64)
    parser.add_argument("-l",   "--loss",
                        type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"])

    # optimizer
    parser.add_argument("-o",   "--optimizer",
                        type=str, default="adam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr",  "--learning_rate",
                        type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",
                        type=float, default=0.0)

    # architecture
    parser.add_argument("-nhl", "--num_layers",
                        type=int, default=3,
                        help="Number of hidden layers")
    parser.add_argument("-sz",  "--hidden_size",
                        type=int, nargs="+", default=[128],
                        help="Neurons per hidden layer. "
                             "One value = same for all layers. "
                             "Or pass one value per layer.")
    parser.add_argument("-a",   "--activation",
                        type=str, default="relu",
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-wi",  "--weight_init",
                        type=str, default="xavier",
                        choices=["random", "xavier"])

    # misc
    parser.add_argument("--val_split",   type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--save_dir",    type=str,   default="models",
                        help="Relative path to save best model weights")
    parser.add_argument("--no_wandb",    action="store_true",
                        help="Disable W&B logging entirely")

    return parser.parse_args()


# ── helpers ────────────────────────────────────────────────────────────────────

def resolve_hidden_sizes(hidden_size: list, num_layers: int) -> list:
    """
    If one value given → broadcast to all layers.
    If num_layers values given → use as-is.
    """
    if len(hidden_size) == 1:
        return [hidden_size[0]] * num_layers
    elif len(hidden_size) == num_layers:
        return hidden_size
    else:
        raise ValueError(
            f"--hidden_size must have 1 value (broadcast) "
            f"or exactly {num_layers} values, got {len(hidden_size)}."
        )


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    # resolve architecture
    hidden_sizes = resolve_hidden_sizes(args.hidden_size, args.num_layers)

    # assignment limits
    assert args.num_layers <= 6,       "Keep num_layers ≤ 6"
    assert max(hidden_sizes) <= 128,   "Keep hidden neurons per layer ≤ 128"

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    config    = {**vars(args), "hidden_sizes": hidden_sizes}

    if use_wandb:
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            config  = config,
        )

    # ── print run summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Dataset    : {args.dataset.upper()}")
    print(f"  Hidden     : {hidden_sizes}")
    print(f"  Activation : {args.activation}   Loss : {args.loss}")
    print(f"  Optimizer  : {args.optimizer}   LR   : {args.learning_rate}")
    print(f"  Epochs     : {args.epochs}       Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    # ── data ─────────────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        args.dataset,
        val_split = args.val_split,
        seed      = args.seed,
    )

    # ── model ────────────────────────────────────────────────────────────────
    model = NeuralNetwork(
        input_size   = X_train.shape[1],
        hidden_sizes = hidden_sizes,
        num_classes  = 10,
        activation   = args.activation,
        weight_init  = args.weight_init,
        loss         = args.loss,
    )

    # ── optimizer ────────────────────────────────────────────────────────────
    optimizer = get_optimizer(
        args.optimizer,
        lr           = args.learning_rate,
        weight_decay = args.weight_decay,
    )

    # ── training loop ────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0

    from utils import get_batches

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for X_batch, y_batch in get_batches(X_train, y_train, args.batch_size):
            y_pred = model.forward(X_batch)
            epoch_loss += model.compute_loss(y_pred, y_batch, args.weight_decay)
            model.backward(y_batch, y_pred, args.weight_decay)
            model.update_weights(optimizer)
            n_batches += 1

        # ── epoch metrics ────────────────────────────────────────────────────
        train_loss = epoch_loss / n_batches
        train_acc  = model.evaluate(X_train, y_train)

        val_pred   = model.forward(X_val)
        val_loss   = model.compute_loss(val_pred, y_val, args.weight_decay)
        val_acc    = model.evaluate(X_val, y_val)

        print(
            f"Epoch [{epoch:>3}/{args.epochs}]  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if use_wandb:
            wandb.log({
                "epoch":      epoch,
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            })

        # ── save best model ──────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(os.path.join(args.save_dir, "best_model.npy"))

            best_config = {
                "dataset":       args.dataset,
                "hidden_sizes":  hidden_sizes,
                "activation":    args.activation,
                "weight_init":   args.weight_init,
                "loss":          args.loss,
                "optimizer":     args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay":  args.weight_decay,
                "batch_size":    args.batch_size,
                "epochs":        args.epochs,
                "best_val_acc":  best_val_acc,
            }
            with open(os.path.join(args.save_dir, "best_config.json"), "w") as f:
                json.dump(best_config, f, indent=2)

    # ── final test evaluation ─────────────────────────────────────────────────
    model.load(os.path.join(args.save_dir, "best_model.npy"))
    test_acc  = model.evaluate(X_test, y_test)
    test_pred = model.forward(X_test)
    test_loss = model.compute_loss(test_pred, y_test)

    print(f"\n{'='*60}")
    print(f"  Final Test Acc  : {test_acc:.4f}")
    print(f"  Final Test Loss : {test_loss:.4f}")
    print(f"{'='*60}\n")

    if use_wandb:
        wandb.log({"test_acc": test_acc, "test_loss": test_loss})
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()