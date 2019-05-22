
import scipy.optimize
import numpy as np
import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import mnac
from ._abstract_recurrent_cell import AbstractRecurrentCell

class SillyReRegualizedLinearMNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, mnac_normalized=False, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.mnac_normalized = mnac_normalized

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(0.25)
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

        if self.mnac_normalized:
            c = torch.std(x)
            x_normalized = x / c
            z_normalized = mnac(x_normalized, W, mode='no-idendity')
            return z_normalized * (c ** torch.sum(W, 1))
        else:
            return mnac(x, W, mode='no-idendity')

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class SillyReRegualizedLinearMNACCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(SillyReRegualizedLinearMNACLayer, input_size, hidden_size, **kwargs)
