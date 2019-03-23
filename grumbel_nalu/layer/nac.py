
import scipy
import numpy as np
import torch

def alu_w_variance(r):
    """Calculates the variance of W.

    Asumming \hat{w} and \hat{m} are sampled from a uniform
    distribution with range [-r, r], this is the variance
    of w = tanh(\hat{w})*sigmoid(\hat{m}).
    """
    return (1 - np.tanh(r) / r) * (r - np.tanh(r / 2)) * (1 / (2 * r))

def alu_w_optimal_r(fan):
    """Computes the optimal Uniform[-r, r] given the fan

    This uses numerical optimization.
    TODO: consider if there is an algebraic solution.
    """
    r, _ = scipy.optimize.bisect(lambda r: fan * alu_w_variance(r) - 1, 0, 6)
    return r

class NACLayer(torch.nn.Module):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.M_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        r_fan_in = alu_w_optimal_r(self.in_features)
        r_fan_out = alu_w_optimal_r(self.out_features)
        r = (r_fan_in + r_fan_out) / 2

        torch.nn.init.uniform(self.W_hat, a=-r, b=r)
        torch.nn.init.uniform(self.M_hat, a=-r, b=r)

    def forward(self, input):
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return torch.nn.functional.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
