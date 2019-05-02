
import scipy.optimize
import numpy as np
import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import mnac
from ._abstract_recurrent_cell import AbstractRecurrentCell

class ReRegualizedLinearMNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)

    def regualizer(self):
         return super().regualizer({
            'W': torch.mean(self.W**2 * (1 - self.W)**2),
            'W-OOB': torch.mean(torch.relu(torch.abs(self.W - 0.5) - 0.5)**2)
        })

    def forward(self, x, reuse=False):
        W = torch.clamp(self.W, 0.0, 1.0)
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)

        return mnac(x, W, mode='prod')

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class ReRegualizedLinearMNACCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(NACLayer, input_size, hidden_size, **kwargs)
