
import torch

from .nac import NACLayer, NACCell
from .gumbel_nac import GumbelNACLayer, GumbelNACCell
from .gumbel_nalu import GumbelNALULayer, GumbelNALUCell
from .nalu import NALULayer, NALUCell
from .basic import BasicLayer
from ..abstract import ExtendedTorchModule

class GeneralizedLayer(ExtendedTorchModule):
    """Abstracts all layers, both basic, NAC and NALU

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
        unit_name: name of the unit (e.g. NAC, Sigmoid, Tanh)
    """

    def __init__(self, in_features, out_features, unit_name, writer=None, **kwags):
        super().__init__('layer', writer=writer, **kwags)
        self.in_features = in_features
        self.out_features = out_features
        self.unit_name = unit_name

        if unit_name == 'NAC':
            self.layer = NACLayer(in_features, out_features,
                                  writer=self.writer,
                                  **kwags)
        elif unit_name == 'GumbelNAC':
            self.layer = GumbelNACLayer(in_features, out_features,
                                        writer=self.writer,
                                        **kwags)
        elif unit_name == 'NALU':
            self.layer = NALULayer(in_features, out_features,
                                   writer=self.writer,
                                   **kwags)
        elif unit_name == 'GumbelNALU':
            self.layer = GumbelNALULayer(in_features, out_features,
                                         writer=self.writer,
                                         **kwags)
        else:
            self.layer = BasicLayer(in_features, out_features,
                                    activation=unit_name,
                                    writer=self.writer,
                                    **kwags)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, input):
        return self.layer(input)

    def extra_repr(self):
        return 'in_features={}, out_features={}, unit_name={}'.format(
            self.in_features, self.out_features, self.unit_name
        )

class GeneralizedCell(ExtendedTorchModule):
    """Abstracts all cell, RNN-tanh, RNN-ReLU, GRU, LSTM, NAC and NALU

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
        unit_name: name of the unit (e.g. RNN-tanh, LSTM, NAC)
    """
    def __init__(self, input_size, hidden_size, unit_name, writer=None, **kwags):
        super().__init__('cell', writer=writer, **kwags)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.unit_name = unit_name

        if unit_name == 'NAC':
            self.cell = NACCell(input_size, hidden_size,
                                writer=self.writer,
                                **kwags)
        elif unit_name == 'GumbelNAC':
            self.cell = GumbelNACCell(input_size, hidden_size,
                                      writer=self.writer,
                                      **kwags)
        elif unit_name == 'NALU':
            self.cell = NALUCell(input_size, hidden_size,
                                 writer=self.writer,
                                 **kwags)
        elif unit_name == 'GumbelNALU':
            self.cell = GumbelNALUCell(input_size, hidden_size,
                                       writer=self.writer,
                                       **kwags)
        elif unit_name == 'GRU':
            self.cell = torch.nn.GRUCell(input_size, hidden_size,
                                          **kwags)
        elif unit_name == 'LSTM':
            self.cell = torch.nn.LSTMCell(input_size, hidden_size,
                                          **kwags)
        elif unit_name == 'RNN-tanh':
            self.cell = torch.nn.RNNCell(input_size, hidden_size,
                                         nonlinearity='tanh',
                                         **kwags)
        elif unit_name == 'RNN-ReLU':
            self.cell = torch.nn.RNNCell(input_size, hidden_size,
                                         nonlinearity='relu',
                                         **kwags)
        else:
            raise NotImplementedError(
                f'{unit_name} is not an implemented cell type')

    def reset_parameters(self):
        self.cell.reset_parameters()

    def forward(self, x_t, h_tm1):
        return self.cell(x_t, h_tm1)

    def extra_repr(self):
        return 'input_size={}, hidden_size={}, unit_name={}'.format(
            self.input_size, self.hidden_size, self.unit_name
        )
