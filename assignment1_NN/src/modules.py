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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization.
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        fan_in = in_features
        fan_out = out_features

        if input_layer:
            # Xavier initialization
            std = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            # Kaiming initialization (for ReLU layers)
            std = np.sqrt(2.0 / fan_in)

        # Initialize weights
        self.params['weight'] = np.random.randn(fan_out, fan_in) * std

        # Initialize biases
        self.params['bias'] = np.zeros((fan_out, 1))

        # Initialize gradients
        self.grads['weight'] = np.zeros((fan_out, fan_in))
        self.grads['bias'] = np.zeros((fan_out, 1))

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = x
        W = self.params['weight']
        b = self.params['bias']
        out = W @ x + b
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.cache  # input from forward pass
        W = self.params['weight']

        # Gradient w.r.t weights
        self.grads['weight'] = dout @ x.T

        # Gradient w.r.t bias
        self.grads['bias'] = np.sum(dout, axis=1, keepdims=True)

        # Gradient w.r.t input (to pass to previous layer)
        dx = W.T @ dout

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = x  # store for backward pass

        out = np.where(
            x > 0,
            x,
            self.alpha * (np.exp(x) - 1)
        )
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.cache

        dx_local = np.where(
                  x > 0,
                  1,
                  self.alpha * np.exp(x)
        )

        dx = dout * dx_local
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Max trick for numerical stability (per column / per sample)
        x_shift = x - np.max(x, axis=0, keepdims=True)

        exp_x = np.exp(x_shift)
        out = exp_x / np.sum(exp_x, axis=0, keepdims=True)

        # store for backward
        self.cache = out

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        s = self.cache  # softmax output (C, N)
        dot = np.sum(dout * s, axis=0, keepdims=True)  # (1, N)
        dx = s * (dout - dot)
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        eps = 1e-12  # voorkomt log(0)
        x_clipped = np.clip(x, eps, 1.0)

        # cache opslaan als je wil (handig)
        self.cache = (x_clipped, y)

        # cross-entropy per sample, dan mean over batch
        loss_per_sample = -np.sum(y * np.log(x_clipped), axis=0)  # shape (N,)
        out = np.mean(loss_per_sample)

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        eps = 1e-12
        x_clipped = np.clip(x, eps, 1.0)

        N = x.shape[1]  # batch_size
        dx = -(y / x_clipped) / N

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx