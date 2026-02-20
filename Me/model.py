import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class MLP(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
        """ Initializing the MLP module """
        super().__init__()
        layers = []
        dims = n_inputs + n_hidden + n_classes

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i+1]

            lin = nn.Linear(fan_in, fan_out, bias=True)
            nn.init.kaiming_normal_(lin.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.zeros_(lin.bias)
            layers.append(lin)

            if i < len(dims) - 2:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(fan_out))
                layers.append(nn.ReLU())


        self.net = nn.Sequential(*layers)


    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.net(x)
        return out
    
    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device