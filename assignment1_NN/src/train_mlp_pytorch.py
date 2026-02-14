################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      llabels: 1D int array of size [batch_size]. Ground truth labels for
               each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Stap 1: voorspelde klasse = index van hoogste logit
    preds = predictions.argmax(dim=1)

    # Stap 2: vergelijk met targets
    correct = (preds == targets)

    # Stap 3: gemiddelde nemen (float)
    accuracy = correct.float().mean().item()

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()  # zet model in evaluation mode (belangrijk voor BatchNorm)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # geen gradients nodig tijdens evaluatie
        for inputs, targets in data_loader:
            outputs = model(inputs)

            preds = outputs.argmax(dim=1)

            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

    avg_accuracy = total_correct / total_samples
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # --- Get train/val/test loaders (works if returned as tuple or dict) ---
    train_loader = cifar10_loader["train"]
    val_loader   = cifar10_loader["validation"]
    test_loader  = cifar10_loader["test"]

    if val_loader is None:
        val_loader = test_loader


    # TODO: Initialize model and loss module
    n_inputs = 3 * 32 * 32
    n_classes = 10

    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=n_classes,
                use_batch_norm=use_batch_norm).to(device)
    loss_module = nn.CrossEntropyLoss()

    # TODO: Do optimization with the simple SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    val_accuracies = []
    # TODO: Training loop including validation
    train_losses = []

    best_val_acc = -1.0
    best_epoch = -1
    best_model = deepcopy(model)

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        n_seen = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # CIFAR10 images -> flatten to vectors
            inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()

            logits = model(inputs)
            loss = loss_module(logits, targets)

            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

        avg_train_loss = running_loss / max(1, n_seen)
        train_losses.append(avg_train_loss)

        # --- Validation accuracy on full validation set (size-weighted) ---
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                inputs = inputs.view(inputs.size(0), -1)

                logits = model(inputs)
                preds = logits.argmax(dim=1)

                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

        val_acc = total_correct / max(1, total_samples)
        val_accuracies.append(val_acc)

        # --- Save best model (by validation accuracy) ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model = deepcopy(model)

    # TODO: Test best model
    best_model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)

            logits = best_model(inputs)
            preds = logits.argmax(dim=1)

            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

    test_accuracy = total_correct / max(1, total_samples)

    # TODO: Add any information you might want to save for plotting
    # --- Logging (you can add more if you want) ---
    logging_dict = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "lr": lr,
        "batch_size": batch_size,
        "hidden_dims": hidden_dims,
        "use_batch_norm": use_batch_norm,
        "seed": seed,
        "device": str(device),
    }

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=5, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    torch.save(model.state_dict(), "best_model.pth")
    print(logging_dict)

    # Feel free to add any additional functions, such as plotting of the loss curve here
