
import torch

from ..abstract import ExtendedTorchModule
from ..functional import mnac
from ._abstract_recurrent_cell import AbstractRecurrentCell

class GumbelMNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features

        # Define the temperature tau
        self.tau = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)

        # Define the target weights. Also, put 0 last such that p1 = p2 = 0
        # corresponds to p3 = 1 => w = 0.
        self.register_buffer('target_weights', torch.tensor([1, -1, 0], dtype=torch.float32))

        # Initialize a tensor, that will be the placeholder for the uniform samples
        self.U = torch.Tensor(out_features, in_features, 3)

        # We will only two parameters per weight, this is to prevent the redundancy
        # there would otherwise exist. This also makes it much more comparable with
        # NAC.
        self.W_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.register_parameter('bias', None)

    def reset_parameters(self):
        # Initialize to zero, the source of randomness can come from the Gumbel sampling.
        torch.nn.init.constant_(self.W_hat, 0)
        torch.nn.init.constant_(self.tau, 1)

    def forward(self, x, reuse=False):
        if self.allow_random:
            gumbel = -torch.log(1e-8 - torch.log(torch.rand(self.out_features, self.in_features, device=x.device) + 1e-8))
            W = torch.sigmoid((self.W_hat + gumbel) / self.tau)
        else:
            W = torch.sigmoid(self.W_hat)

        # Compute the linear multiplication as usual
        expected_W = torch.sigmoid(self.W_hat)
        self.writer.add_histogram('W', expected_W)
        self.writer.add_tensor('W', expected_W, verbose_only=False)

        return mnac(x, W)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class GumbelMNACCell(AbstractRecurrentCell):
    """Implements the Gumbel NAC (Gumbel Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(GumbelNACLayer, input_size, hidden_size, **kwargs)
