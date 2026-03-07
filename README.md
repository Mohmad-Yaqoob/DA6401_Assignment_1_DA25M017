# DA6401 --- Assignment 1: Multi‑Layer Perceptron (NumPy Implementation)

Implementation of a **fully configurable Multi‑Layer Perceptron (MLP)**
from scratch using **pure NumPy**.\
The model supports multiple optimizers, activation functions, and weight
initialization strategies and is evaluated on **MNIST and
Fashion‑MNIST** datasets.

All experiments and visualizations are logged using **Weights & Biases
(W&B)**.

------------------------------------------------------------------------

## Project Structure

``` text
da6401_assignment_1/
│
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py         # Activation functions (sigmoid, tanh, relu, softmax)
│   │   ├── neural_layer.py        # Fully connected layer implementation
│   │   ├── neural_network.py      # Main MLP model
│   │   ├── objective_functions.py # Cross‑Entropy and MSE losses
│   │   └── optimizers.py          # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py         # MNIST / Fashion‑MNIST dataset loader
│   │
│   ├── train.py                   # Model training script
│   └── inference.py               # Evaluation script
│
├── notebooks/
│   └── wandb_demo.ipynb           # W&B experiments (Q2.1 – Q2.10)
│
├── models/                        # Saved model weights
├── sweep_config.yaml              # W&B hyperparameter sweep configuration
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## Setup

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Train the Model

Example training command:

``` bash
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

------------------------------------------------------------------------

## Run Inference

``` bash
cd src
python inference.py --no_wandb
```

This evaluates the trained model on the test dataset.

------------------------------------------------------------------------

## Hyperparameter Sweep (W&B)

Run automated hyperparameter search:

``` bash
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

------------------------------------------------------------------------

## CLI Arguments

  -----------------------------------------------------------------------
  Flag          Description                         Default
  ------------- ----------------------------------- ---------------------
  `-d`          Dataset (`mnist` or                 `mnist`
                `fashion_mnist`)                    

  `-e`          Number of epochs                    `10`

  `-b`          Batch size                          `64`

  `-l`          Loss function (`cross_entropy` or   `cross_entropy`
                `mse`)                              

  `-o`          Optimizer (`sgd`, `momentum`,       `adam`
                `nag`, `rmsprop`, `adam`, `nadam`)  

  `-lr`         Learning rate                       `0.001`

  `-wd`         Weight decay (L2 regularization)    `0.0`

  `-nhl`        Number of hidden layers             `3`

  `-sz`         Hidden layer sizes                  `[128]`

  `-a`          Activation (`sigmoid`, `tanh`,      `relu`
                `relu`)                             

  `-wi`         Weight initialization (`random`,    `xavier`
                `xavier`)                           
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Experiments (W&B)

All experiments required in the assignment were implemented and logged
using **Weights & Biases**.

Experiments performed:

-   Data exploration
-   Hyperparameter sweeps
-   Optimizer comparison
-   Vanishing gradient analysis
-   Dead neuron investigation
-   Loss function comparison
-   Global performance analysis
-   Error analysis
-   Weight initialization study
-   Fashion‑MNIST transfer challenge

### W&B Report

https://api.wandb.ai/links/da25m017-indian-institute-of-technology-madras/esjzo97n

------------------------------------------------------------------------

## Results Summary

### MNIST Best Configuration

    Architecture : 3 Hidden Layers (128 neurons each)
    Activation   : ReLU
    Optimizer    : Adam
    Weight Init  : Xavier
    Loss         : Cross‑Entropy
    Validation Accuracy ≈ 97.8%

------------------------------------------------------------------------

### Fashion‑MNIST Transfer Experiment

  --------------------------------------------------------------------------
  Configuration    Architecture   Optimizer    Activation   Test Accuracy
  ---------------- -------------- ------------ ------------ ----------------
  Config A         3×128          Adam         ReLU         **0.8884**

  Config B         4×128          Nadam        ReLU         **0.8851**

  Config C         3×128          Adam         Tanh         **0.8848**
  
  --------------------------------------------------------------------------

  

The **Adam + ReLU configuration that performed best on MNIST also
achieved the highest accuracy on Fashion‑MNIST**, although overall
accuracy is lower due to the increased complexity of clothing images.

------------------------------------------------------------------------

## Author

**Mohmad Yaqoob**\
M.Tech --- Data Science & AI\
IIT Madras

Roll No: **DA25M017**
