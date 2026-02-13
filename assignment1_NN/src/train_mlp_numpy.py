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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # stap 1: pak de voorspelde class (index van grootste logit)
    pred_classes = np.argmax(predictions, axis=1)

    # stap 2: vergelijk met echte labels (True/False array)
    correct = pred_classes == targets

    # stap 3: gemiddelde nemen (True = 1, False = 0)
    accuracy = np.mean(correct)
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
    total_correct = 0
    total_samples = 0

    for inputs, targets in data_loader:

        # forward pass
        outputs = model.forward(inputs)

        # predicted classes
        pred_classes = np.argmax(outputs, axis=1)

        # aantal correcte voorspellingen in deze batch
        correct = np.sum(pred_classes == targets)

        # update totals
        total_correct += correct
        total_samples += targets.shape[0]

    # gemiddelde accuracy over hele dataset
    avg_accuracy = total_correct / total_samples
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
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

    cifar10 = cifar10_utils.get_cifar10(data_dir)
    loaders = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size, return_numpy=True)

    train_loader = loaders["train"]
    val_loader   = loaders["validation"]   # <-- FIX
    test_loader  = loaders["test"]

    model = MLP(n_inputs=32*32*3, n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()

    # --- save/restore best weights (no deepcopy) ---
    def get_model_state(m):
        state = []
        for layer in m.layers:  # pas aan als jouw lijst anders heet
            if hasattr(layer, "W") and hasattr(layer, "b"):
                state.append((layer.W.copy(), layer.b.copy()))
            else:
                state.append(None)
        return state

    def set_model_state(m, state):
        for layer, saved in zip(m.layers, state):
            if saved is None:
                continue
            W, b = saved
            layer.W = W.copy()
            layer.b = b.copy()

    train_losses = []
    val_accuracies = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        epoch_num_samples = 0

        for X, y in train_loader:
            # flatten if needed
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)

            logits = model.forward(X)
            loss = loss_module.forward(logits, y)

            dlogits = loss_module.backward(logits, y)
            model.backward(dlogits)
            model.step(lr)

            bs = y.shape[0]
            epoch_loss_sum += float(loss) * bs
            epoch_num_samples += bs

        avg_train_loss = epoch_loss_sum / epoch_num_samples
        train_losses.append(avg_train_loss)

        val_acc = evaluate_model(model, val_loader)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = get_model_state(model)

        print(f"Epoch {epoch+1}/{epochs} | train_loss={avg_train_loss:.4f} | val_acc={val_acc:.4f}")

    # load best weights
    if best_state is not None:
        set_model_state(model, best_state)

    test_accuracy = evaluate_model(model, test_loader)

    logging_dict = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_accuracy,
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

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
