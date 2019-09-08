
import math
import torch

from ..functional import sparsity_error
from ..abstract import ExtendedTorchModule
from ._abstract_recurrent_cell import AbstractRecurrentCell

ACTIVATIONS = {
    'Tanh': torch.tanh,
    'Sigmoid': torch.sigmoid,
    'ReLU6': torch.nn.functional.relu6,
    'Softsign': torch.nn.functional.softsign,
    'SELU': torch.selu,
    'ELU': torch.nn.functional.elu,
    'ReLU': torch.relu,
    'linear': lambda x: x
}

INITIALIZATIONS = {
    'Tanh': lambda W: torch.nn.init.xavier_uniform_(
        W, gain=torch.nn.init.calculate_gain('tanh')),

    'Sigmoid': lambda W: torch.nn.init.xavier_uniform_(
        W, gain=torch.nn.init.calculate_gain('sigmoid')),

    'ReLU6': lambda W: torch.nn.init.kaiming_uniform_(
        W, nonlinearity='relu'),

    'Softsign': lambda W: torch.nn.init.xavier_uniform_(
        W, gain=1),

    'SELU': lambda W: torch.nn.init.uniform_(
        W, a=-math.sqrt(3/W.size(1)), b=math.sqrt(3/W.size(1))),

    # ELU: The weights have been initialized according to (He et al., 2015).
    #      source: https://arxiv.org/pdf/1511.07289.pdf
    'ELU': lambda W: torch.nn.init.kaiming_uniform_(
        W, nonlinearity='relu'),

    'ReLU': lambda W: torch.nn.init.kaiming_uniform_(
        W, nonlinearity='relu'),

    'linear': lambda W: torch.nn.init.xavier_uniform_(
        W, gain=torch.nn.init.calculate_gain('linear'))
}

class BasicLayer(ExtendedTorchModule):
    ACTIVATIONS = set(ACTIVATIONS.keys())

    def __init__(self, in_features, out_features, activation='linear', bias=True, **kwargs):
        super().__init__('basic', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        if activation not in ACTIVATIONS:
            raise NotImplementedError(
                f'the activation {activation} is not implemented')

        self.activation_fn = ACTIVATIONS[activation]
        self.initializer = INITIALIZATIONS[activation]

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        self.initializer(self.W)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, reuse=False):
        self.writer.add_histogram('W', self.W)
        self.writer.add_tensor('W', self.W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(self.W), verbose_only=False)
        return self.activation_fn(
            torch.nn.functional.linear(input, self.W, self.bias)
        )

    def extra_repr(self):
        return 'in_features={}, out_features={}, activation={}'.format(
            self.in_features, self.out_features, self.activation
        )

class BasicCell(AbstractRecurrentCell):
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(BasicLayer, input_size, hidden_size, **kwargs)
