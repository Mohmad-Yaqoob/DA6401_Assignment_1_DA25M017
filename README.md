# DA6401 — Assignment 1: Multi-Layer Perceptron

Pure NumPy implementation of a configurable MLP trained on MNIST and Fashion-MNIST.

## Project Structure
```
da6401_assignment_1/
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py         # sigmoid, tanh, relu, softmax
│   │   ├── neural_layer.py        # single dense layer
│   │   ├── neural_network.py      # full MLP
│   │   ├── objective_functions.py # cross-entropy, MSE
│   │   └── optimizers.py          # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py         # MNIST/Fashion-MNIST loader
│   ├── train.py                   # training script
│   └── inference.py               # evaluation script
├── notebooks/
│   └── wandb_demo.ipynb           # all W&B experiments Q2.1-Q2.10
├── models/                        # saved weights (auto-created)
├── sweep_config.yaml              # W&B hyperparameter sweep
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
cd src
python train.py \
  -d mnist \
  -e 15 \
  -b 64 \
  -l cross_entropy \
  -o adam \
  -lr 0.001 \
  -wd 0.0 \
  -nhl 3 \
  -sz 128 128 128 \
  -a relu \
  -wi xavier
```

## Inference
```bash
cd src
python inference.py --no_wandb
```

## Hyperparameter Sweep
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

## CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-d` | Dataset: `mnist` or `fashion_mnist` | `mnist` |
| `-e` | Epochs | `10` |
| `-b` | Batch size | `64` |
| `-l` | Loss: `cross_entropy` or `mse` | `cross_entropy` |
| `-o` | Optimizer: `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` | `adam` |
| `-lr` | Learning rate | `0.001` |
| `-wd` | Weight decay (L2) | `0.0` |
| `-nhl` | Number of hidden layers | `3` |
| `-sz` | Neurons per hidden layer | `[128]` |
| `-a` | Activation: `sigmoid`, `tanh`, `relu` | `relu` |
| `-wi` | Weight init: `random` or `xavier` | `xavier` |

## W&B Report

> Link: *(add your public W&B report link here)*

## Author

Name: *Mohmad Yaqoob*  
Roll No: DA25M017