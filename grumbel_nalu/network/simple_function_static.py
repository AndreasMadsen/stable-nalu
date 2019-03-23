
import torch
from ..layer import GeneralizedLayer

class SimpleFunctionStaticNetwork(torch.nn.Module):
    def __init__(self, unit_name, input_size=100):
        super().__init__()

        self.unit_name = unit_name
        self.input_size = input_size

        self.layer_1 = GeneralizedLayer(input_size, 2, unit_name)
        self.layer_2 = GeneralizedLayer(2, 1, unit_name)

    def reset_parameters(self):
        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()

    def forward(self, input):
        z_1 = self.layer_1.forward(input)
        z_2 = self.layer_2.forward(z_1)
        return z_2

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )
