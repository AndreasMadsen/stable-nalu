
import math
import torch
import numpy as np
import scipy.optimize

from ..functional import mnac, sparsity_error
from ..abstract import ExtendedTorchModule
from ._abstract_recurrent_cell import AbstractRecurrentCell

class MNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features

        self.W_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = math.sqrt(3.0) * std
        torch.nn.init.uniform_(self.W_hat, - r, r)

    def forward(self, x, reuse=False):
        W = torch.sigmoid(self.W_hat)
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)


        return mnac(x, W)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class MNACCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(MNACLayer, input_size, hidden_size, **kwargs)
