
import torch

from ..writer import DummyWriter

class AbstractNALULayer(torch.nn.Module):
    """Implements the NALU (Neural Arithmetic Logic Unit)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, NACOp, in_features, out_features, eps=1e-7, bias=False, writer=DummyWriter()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.writer = writer

        self.nac = NACOp(in_features, out_features, writer=writer.namespace('nac'))
        self.G = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
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
            torch.log(torch.abs(x) + self.eps), reuse=True
        ))
        # y = g (*) a + (1 - g) (*) m
        y = g * a + (1 - g) * m

        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}, eps={}'.format(
            self.in_features, self.out_features, self.eps
        )