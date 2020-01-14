
import scipy.optimize
import numpy as np
import torch

from ..functional import nac_weight, sparsity_error, RegualizerNAUZ
from ..abstract import ExtendedTorchModule
from ._abstract_recurrent_cell import AbstractRecurrentCell

def nac_w_variance(r):
    """Calculates the variance of W.

    Asumming \hat{w} and \hat{m} are sampled from a uniform
    distribution with range [-r, r], this is the variance
    of w = tanh(\hat{w})*sigmoid(\hat{m}).
    """
    if (r == 0):
        return 0
    else:
        return (1 - np.tanh(r) / r) * (r - np.tanh(r / 2)) * (1 / (2 * r))

def nac_w_optimal_r(fan_in, fan_out):
    """Computes the optimal Uniform[-r, r] given the fan

    This uses numerical optimization.
    TODO: consider if there is an algebraic solution.
    """
    fan = max(fan_in + fan_out, 5)
    r = scipy.optimize.bisect(lambda r: fan * nac_w_variance(r) - 2, 0, 10)
    return r

class NACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 regualizer_z = 0, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features

        self.W_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.M_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

        self._regualizer_nau_z = RegualizerNAUZ(
            zero=regualizer_z == 0
        )

    def reset_parameters(self):
        r = nac_w_optimal_r(self.in_features, self.out_features)
        torch.nn.init.uniform_(self.W_hat, a=-r, b=r)
        torch.nn.init.uniform_(self.M_hat, a=-r, b=r)

    def optimize(self, loss):
        self._regualizer_nau_z.reset()

    def regualizer(self):
        W = nac_weight(self.W_hat, self.M_hat, mode='normal')

        return super().regualizer({
            'z': self._regualizer_nau_z(W)
        })

    def forward(self, x, reuse=False):
        if self.allow_random:
            self._regualizer_nau_z.append_input(x)

        W = nac_weight(self.W_hat, self.M_hat, mode='normal')
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)

        return torch.nn.functional.linear(x, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class NACCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(NACLayer, input_size, hidden_size, **kwargs)
