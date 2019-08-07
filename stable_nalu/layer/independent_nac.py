
import torch

from ..functional import nac_weight, sparsity_error
from .nac import NACLayer
from ._abstract_recurrent_cell import AbstractRecurrentCell

class IndependentNACLayer(NACLayer):
    def forward(self, input, reuse=False):
        W = nac_weight(self.W_hat, self.M_hat, mode='independent')
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)

        return torch.nn.functional.linear(input, W, self.bias)

class IndependentNACCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(IndependentNACLayer, input_size, hidden_size, **kwargs)
