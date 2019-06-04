
import scipy.optimize
import numpy as np
import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import Regualizer
from ._abstract_recurrent_cell import AbstractRecurrentCell

class ReRegualizedLinearNACLayer(ExtendedTorchModule):
    """Implements the RegualizedLinearNAC

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared',
                 **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.nac_oob = nac_oob

        self._regualizer_bias = Regualizer(
            support='nac', type='bias',
            shape=regualizer_shape
        )
        self._regualizer_oob = Regualizer(
            support='nac', type='oob',
            shape=regualizer_shape,
            zero=self.nac_oob == 'clip'
        )

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        r = min(0.5, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, -r, r)

    def optimize(self, loss):
        if self.nac_oob == 'clip':
            self.W.data.clamp_(-1.0, 1.0)

    def regualizer(self):
         return super().regualizer({
            'W': self._regualizer_bias(self.W),
            'W-OOB': self._regualizer_oob(self.W)
        })

    def forward(self, input, reuse=False):
        W = torch.clamp(self.W, -1.0, 1.0)
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        return torch.nn.functional.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class ReRegualizedLinearNACCell(AbstractRecurrentCell):
    """Implements the RegualizedLinearNAC as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(ReRegualizedLinearNACLayer, input_size, hidden_size, **kwargs)
