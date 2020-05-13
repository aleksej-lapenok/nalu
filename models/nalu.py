import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .nac import NeuralAccumulatorCell
from torch.nn.parameter import Parameter


class NeuralArithmeticLogicUnitCell(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 1e-8

        self.G = Parameter(torch.zeros([out_dim, in_dim], dtype=torch.float, device=device))
        self.nac = NeuralAccumulatorCell(in_dim, out_dim, device)
        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.G, a=math.sqrt(5))

    def forward(self, input):
        a = self.nac(input)
        g = torch.sigmoid(F.linear(input, self.G, self.bias))
        add_sub = g * a
        log_input = torch.log(torch.abs(input) + self.eps)
        m = torch.exp(self.nac(log_input))
        mul_div = (1 - g) * m
        y = add_sub + mul_div
        return y

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )


class NALU(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, device):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        layers = []
        for i in range(num_layers):
            layers.append(
                NeuralArithmeticLogicUnitCell(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < num_layers - 1 else out_dim,
                    device
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out
