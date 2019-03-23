
import math
import torch

ACTIVATIONS = {
    'Tanh': torch.nn.functional.tanh,
    'Sigmoid': torch.nn.functional.sigmoid,
    'ReLU6': torch.nn.functional.relu6,
    'Softsign': torch.nn.functional.softsign,
    'SELU': torch.nn.functional.selu,
    'ELU': torch.nn.functional.elu,
    'ReLU': torch.nn.functional.relu,
    'linear': lambda x: x
}

INITIALIZATIONS = {
    'Tanh': lambda W: torch.nn.init.xavier_uniform_(W, gain=torch.nn.init.calculate_gain('tanh')),
    'Sigmoid': lambda W: torch.nn.init.xavier_uniform_(W, gain=torch.nn.init.calculate_gain('sigmoid')),
    'ReLU6': lambda W: torch.nn.init.kaiming_uniform_(W, nonlinearity='relu'),
    'Softsign': lambda W: torch.nn.init.xavier_uniform_(W, gain=1),
    'SELU': lambda W: torch.nn.init.uniform_(W, a=-math.sqrt(3/W.size(1)), b=math.sqrt(3/W.size(1))),
    # ELU: The weights have been initialized according to (He et al., 2015).
    #      source: https://arxiv.org/pdf/1511.07289.pdf
    'ELU': lambda W: torch.nn.init.kaiming_uniform_(W, nonlinearity='relu'),
    'ReLU': lambda W: torch.nn.init.kaiming_uniform_(W, nonlinearity='relu'),
    'linear': lambda W: torch.nn.init.xavier_uniform_(W, gain=torch.nn.init.calculate_gain('linear'))
}

class BasicLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, activation='linear'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        if activation not in ACTIVATIONS:
            raise NotImplementedError(f'the activation {activation} is not implemented')

        self.activation_fn = ACTIVATIONS[activation]
        self.initializer = INITIALIZATIONS[activation]

        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.initializer(self.weight)

    def forward(self, input):
        return self.activation_fn(
            torch.nn.functional.linear(input, self.weight, self.bias)
        )

    def extra_repr(self):
        return 'in_features={}, out_features={}, activation={}'.format(
            self.in_features, self.out_features, self.activation
        )
