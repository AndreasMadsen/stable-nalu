
import torch
from ..layer import GeneralizedLayer, GeneralizedCell
from ..writer import DummyWriter

class SimpleFunctionRecurrentNetwork(torch.nn.Module):
    def __init__(self, unit_name, input_size=10,
                 writer=DummyWriter(), **kwags):
        super().__init__()

        self.unit_name = unit_name
        self.input_size = input_size
        self.hidden_size = 2

        # Since for the 'mul' problem, the zero_state should be 1, and for the
        # 'add' problem it should be 0. The zero_states are allowed to be
        # # optimized.
        if unit_name == 'LSTM':
            self.zero_state = torch.nn.ParameterDict({
                'h_t0': torch.nn.Parameter(torch.Tensor(self.hidden_size)),
                'c_t0': torch.nn.Parameter(torch.Tensor(self.hidden_size))
            })
        else:
            self.zero_state = torch.nn.Parameter(torch.Tensor(self.hidden_size))

        self.recurent_cell = GeneralizedCell(input_size, self.hidden_size,
                                             unit_name,
                                             writer=writer.namespace('recurrent_layer'),
                                             **kwags)
        self.output_layer = GeneralizedLayer(self.hidden_size, 1,
                                             unit_name if unit_name in {'NAC', 'NALU'} else 'linear',
                                             writer=writer.namespace('output_layer'),
                                             **kwags)

    def reset_parameters(self):
        if self.unit_name == 'LSTM':
            for zero_state in self.zero_state.values():
                torch.nn.init.zeros_(zero_state)
        else:
            torch.nn.init.zeros_(self.zero_state)

        self.recurent_cell.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x):
        """Performs recurrent iterations over the input.

        Arguments:
            input: Expected to have the shape [obs, time, dims]
        """
        # Perform recurrent iterations over the input
        if self.unit_name == 'LSTM':
            h_tm1 = tuple(zero_state.repeat(x.size(0), 1) for zero_state in self.zero_state.values())
        else:
            h_tm1 = self.zero_state.repeat(x.size(0), 1)

        for t in range(x.size(1)):
            x_t = x[:, t]
            h_t = self.recurent_cell(x_t, h_tm1)
            h_tm1 = h_t

        # Grap the final hidden output and use as the output from the recurrent layer
        z_1 = h_t[0] if self.unit_name == 'LSTM' else h_t
        z_2 = self.output_layer(z_1)
        return z_2

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )
