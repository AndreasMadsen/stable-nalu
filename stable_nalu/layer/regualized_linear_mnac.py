
import scipy.optimize
import numpy as np
import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import mnac, sparsity_error
from ._abstract_recurrent_cell import AbstractRecurrentCell

class RegualizedLinearMNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 regualizer_shape='squared',
                 mnac_epsilon=0, mnac_normalized=False, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.mnac_normalized = mnac_normalized

        self._regualizer_bias = Regualizer(
            support='mnac', type='bias',
            shape=regualizer_shape, zero_epsilon=mnac_epsilon
        )

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = math.sqrt(3.0) * std
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)

    def regualizer(self):
         return super().regualizer({
            'W': self._regualizer_bias(self.W)
        })

    def forward(self, x, reuse=False):
        W = self.W
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)


        if self.mnac_normalized:
            c = torch.std(x)
            x_normalized = x / c
            z_normalized = mnac(x_normalized, W, mode='prod')
            return z_normalized * (c ** torch.sum(W, 1))
        else:
            return mnac(x, W, mode='prod')

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class RegualizedLinearMNACCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(ReRegualizedLinearMNACLayer, input_size, hidden_size, **kwargs)
