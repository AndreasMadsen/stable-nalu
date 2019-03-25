
import scipy.optimize
import numpy as np
import torch

from .nac import NACLayer
from ..writer import DummyWriter

class NALULayer(torch.nn.Module):
    """Implements the NALU (Neural Arithmetic Logic Unit)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, eps=1e-7, writer=DummyWriter()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.writer = writer

        self.nac = NACLayer(in_features, out_features, writer=writer.namespace('nac'))
        self.G = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        self.nac.reset_parameters()
        torch.nn.init.xavier_uniform_(
            self.G,
            gain=torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, x):
        # g = sigmoid(G x)
        g = torch.sigmoid(torch.nn.functional.linear(x, self.G, self.bias))
        self.writer.add_summary('gate', g)
        # a = W x = nac(x)
        a = self.nac(x)
        # m = exp(W log(|x| + eps)) = exp(nac(log(|x| + eps)))
        m = torch.exp(self.nac(
            torch.log(torch.abs(x) + self.eps)
        ))
        # y = g (*) a + (1 - g) (*) m
        y = g * a + (1 - g) * m

        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}, eps={}'.format(
            self.in_features, self.out_features, self.eps
        )

class NALUCell(torch.nn.Module):
    """Implements the NALU (Neural Arithmetic Logic Unit) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, writer=DummyWriter()):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nalu = NACLayer(input_size + hidden_size, hidden_size, writer=writer)

    def reset_parameters(self):
        self.nac.reset_parameters()

    def forward(self, x_t, h_tm1):
        return self.nalu(torch.cat((x_t, h_tm1), dim=1))

    def extra_repr(self):
        return 'input_size={}, hidden_size={}'.format(
            self.input_size, self.hidden_size
        )
