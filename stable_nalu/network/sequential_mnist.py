
import torch
from ..abstract import ExtendedTorchModule
from ..layer import GeneralizedLayer, GeneralizedCell
from .regression_mnist import RegressionMnisNetwork

class SequentialMnistNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedCell.UNIT_NAMES

    def __init__(self, unit_name, output_size, writer=None, nac_mul='none', eps=1e-7, **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name
        self.output_size = output_size
        self.nac_mul = nac_mul
        self.eps = eps

        # TODO: maybe don't make them learnable, properly zero will surfise here
        if unit_name == 'LSTM':
            self.zero_state_h = torch.nn.Parameter(torch.Tensor(self.output_size))
            self.zero_state_c = torch.nn.Parameter(torch.Tensor(self.output_size))
        else:
            self.zero_state = torch.nn.Parameter(torch.Tensor(self.output_size))

        self.image2label = RegressionMnisNetwork()

        if nac_mul == 'mnac':
            unit_name = unit_name[0:-3] + 'MNAC'
        if unit_name == 'LSTM':
            del kwags['nalu_bias']
            del kwags['nalu_two_nac']
            del kwags['nalu_two_gate']
            del kwags['nalu_mul']
            del kwags['nalu_gate']
        self.recurent_cell = GeneralizedCell(1, self.output_size,
                                             unit_name,
                                             writer=self.writer,
                                             **kwags)
        self.reset_parameters()

    def reset_parameters(self):
        if self.unit_name == 'LSTM':
            torch.nn.init.zeros_(self.zero_state_h)
            torch.nn.init.zeros_(self.zero_state_c)
        else:
            torch.nn.init.zeros_(self.zero_state)

        self.image2label.reset_parameters()
        self.recurent_cell.reset_parameters()

    def forward(self, x):
        """Performs recurrent iterations over the input.

        Arguments:
            input: Expected to have the shape [obs, time, channels=1, width, height]
        """
        # Perform recurrent iterations over the input
        if self.unit_name == 'LSTM':
            h_tm1 = (
                self.zero_state_h.repeat(x.size(0), 1),
                self.zero_state_c.repeat(x.size(0), 1)
            )
            self.writer.add_tensor('h_tm1/h', h_tm1[0])
            self.writer.add_tensor('h_tm1/c', h_tm1[1])
        else:
            h_tm1 = self.zero_state.repeat(x.size(0), 1)
            self.writer.add_tensor('h_tm1', h_tm1)

        for t in range(x.size(1)):
            x_t = x[:, t]
            l_t = self.image2label(x_t)

            if self.nac_mul == 'none' or self.nac_mul == 'mnac':
                h_t = self.recurent_cell(l_t, h_tm1)
            elif self.nac_mul == 'normal':
                h_t = torch.exp(self.recurent_cell(
                    torch.log(torch.abs(l_t) + self.eps),
                    torch.log(torch.abs(h_tm1) + self.eps)
                ))
            h_tm1 = h_t

        # Grap the final hidden output and use as the output from the recurrent layer
        z_1 = h_t[0] if self.unit_name == 'LSTM' else h_t

        return l_t, z_1

    def extra_repr(self):
        return 'unit_name={}, output_size={}'.format(
            self.unit_name, self.output_size
        )
