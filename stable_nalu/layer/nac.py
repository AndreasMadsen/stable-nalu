
import scipy.optimize
import numpy as np
import torch

from ..writer import DummyWriter

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

class NACLayer(torch.nn.Module):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, writer=DummyWriter()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.writer = writer

        self.W_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.M_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        r = nac_w_optimal_r(self.in_features, self.out_features)
        torch.nn.init.uniform_(self.W_hat, a=-r, b=r)
        torch.nn.init.uniform_(self.M_hat, a=-r, b=r)

    def forward(self, input):
        tanh_w_hat = torch.tanh(self.W_hat)
        sigmoid_m_hat = torch.sigmoid(self.M_hat)

        self.writer.add_summary('w_hat_factor', (1 - tanh_w_hat*tanh_w_hat)*sigmoid_m_hat)
        self.writer.add_summary('m_hat_factor', tanh_w_hat*sigmoid_m_hat*(1-sigmoid_m_hat))

        W = tanh_w_hat * sigmoid_m_hat
        return torch.nn.functional.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
