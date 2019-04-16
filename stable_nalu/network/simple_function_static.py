
import torch
from ..abstract import ExtendedTorchModule
from ..layer import GeneralizedLayer, BasicLayer

class SimpleFunctionStaticNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedLayer.UNIT_NAMES

    def __init__(self, unit_name, input_size=100, writer=None, **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name
        self.input_size = input_size

        self.layer_1 = GeneralizedLayer(input_size, 2,
                                        unit_name,
                                        writer=self.writer,
                                        name='layer_1',
                                        **kwags)
        self.layer_2 = GeneralizedLayer(2, 1,
                                        'linear' if unit_name in BasicLayer.ACTIVATIONS else unit_name,
                                        writer=self.writer,
                                        name='layer_2',
                                        **kwags)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()

    def forward(self, input):
        self.writer.add_summary('x', input)
        z_1 = self.layer_1(input)
        self.writer.add_summary('z_1', z_1)
        z_2 = self.layer_2(z_1)
        self.writer.add_summary('z_2', z_2)
        return z_2

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )
