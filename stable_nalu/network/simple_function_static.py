
import torch
from ..abstract import ExtendedTorchModule
from ..layer import GeneralizedLayer, BasicLayer

class SimpleFunctionStaticNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedLayer.UNIT_NAMES

    def __init__(self, unit_name, input_size=100, writer=None, nac_mul='none', eps=1e-7, **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name
        self.input_size = input_size
        self.nac_mul = nac_mul
        self.eps = eps

        self.layer_1 = GeneralizedLayer(input_size, 2,
                                        unit_name,
                                        writer=self.writer,
                                        name='layer_1',
                                        eps=eps, **kwags)
        self.layer_2 = GeneralizedLayer(2, 1,
                                        'linear' if unit_name in BasicLayer.ACTIVATIONS else unit_name,
                                        writer=self.writer,
                                        name='layer_2',
                                        eps=eps, **kwags)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()

    def forward(self, input):
        self.writer.add_summary('x', input)
        z_1 = self.layer_1(input)
        self.writer.add_summary('z_1', z_1)

        if self.nac_mul == 'none':
            z_2 = self.layer_2(z_1)
        elif self.nac_mul == 'normal':
            z_2 = torch.exp(self.layer_2(torch.log(torch.abs(z_1) + self.eps)))
        elif self.nac_mul == 'safe':
            z_2 = torch.exp(self.layer_2(torch.log(torch.abs(z_1 - 1) + 1)))
        else:
            raise ValueError(f'Unsupported nac_mul option ({self.nac_mul})')

        self.writer.add_summary('z_2', z_2)
        return z_2

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )
