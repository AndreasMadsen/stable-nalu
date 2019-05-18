
import torch

from .basic import BasicLayer, BasicCell

from .nac import NACLayer, NACCell
from .mnac import MNACLayer, MNACCell
from .nalu import NALULayer, NALUCell

from .pos_nac import PosNACLayer, PosNACCell
from .pos_nalu import PosNALULayer, PosNALUCell

from .gumbel_nac import GumbelNACLayer, GumbelNACCell
from .gumbel_mnac import GumbelMNACLayer, GumbelMNACCell
from .gumbel_nalu import GumbelNALULayer, GumbelNALUCell

from .linear_nac import LinearNACLayer, LinearNACCell
from .linear_nalu import LinearNALULayer, LinearNALUCell

from .softmax_nac import SoftmaxNACLayer, SoftmaxNACCell
from .softmax_nalu import SoftmaxNALULayer, SoftmaxNALUCell

from .independent_nac import IndependentNACLayer, IndependentNACCell
from .independent_nalu import IndependentNALULayer, IndependentNALUCell

from .hard_softmax_nac import HardSoftmaxNACLayer, HardSoftmaxNACCell
from .hard_softmax_nalu import HardSoftmaxNALULayer, HardSoftmaxNALUCell

from .gradient_bandit_nac import GradientBanditNACLayer, GradientBanditNACCell
from .gradient_bandit_nalu import GradientBanditNALULayer, GradientBanditNALUCell

from .regualized_linear_nac import RegualizedLinearNACLayer, RegualizedLinearNACCell
from .regualized_linear_mnac import RegualizedLinearMNACLayer, RegualizedLinearMNACCell
from .regualized_linear_nalu import RegualizedLinearNALULayer, RegualizedLinearNALUCell

from .re_regualized_linear_nac import ReRegualizedLinearNACLayer, ReRegualizedLinearNACCell
from .re_regualized_linear_mnac import ReRegualizedLinearMNACLayer, ReRegualizedLinearMNACCell
from .re_regualized_linear_nalu import ReRegualizedLinearNALULayer, ReRegualizedLinearNALUCell

from .silly_re_regualized_linear_mnac import SillyReRegualizedLinearMNACLayer, SillyReRegualizedLinearMNACCell

from ..abstract import ExtendedTorchModule

unit_name_to_layer_class = {
    'NAC': NACLayer,
    'MNAC': MNACLayer,
    'NALU': NALULayer,

    'PosNAC': PosNACLayer,
    'PosNALU': PosNALULayer,

    'GumbelNAC': GumbelNACLayer,
    'GumbelMNAC': GumbelMNACLayer,
    'GumbelNALU': GumbelNALULayer,

    'LinearNAC': LinearNACLayer,
    'LinearNALU': LinearNALULayer,

    'SoftmaxNAC': SoftmaxNACLayer,
    'SoftmaxNALU': SoftmaxNALULayer,

    'IndependentNAC': IndependentNACLayer,
    'IndependentNALU': IndependentNALULayer,

    'HardSoftmaxNAC': HardSoftmaxNACLayer,
    'HardSoftmaxNALU': HardSoftmaxNALULayer,

    'GradientBanditNAC': GradientBanditNACLayer,
    'GradientBanditNALU': GradientBanditNALULayer,

    'RegualizedLinearNAC': RegualizedLinearNACLayer,
    'RegualizedLinearMNAC': RegualizedLinearMNACLayer,
    'RegualizedLinearNALU': RegualizedLinearNALULayer,

    'ReRegualizedLinearNAC': ReRegualizedLinearNACLayer,
    'ReRegualizedLinearMNAC': ReRegualizedLinearMNACLayer,
    'ReRegualizedLinearNALU': ReRegualizedLinearNALULayer,

    'SillyReRegualizedLinearNAC': None,
    'SillyReRegualizedLinearMNAC': SillyReRegualizedLinearMNACLayer,
}

unit_name_to_cell_class = {
    'NAC': NACCell,
    'MNAC': MNACCell,
    'NALU': NALUCell,

    'GumbelNAC': GumbelNACCell,
    'GumbelMNAC': GumbelMNACCell,
    'GumbelNALU': GumbelNALUCell,

    'SoftmaxNAC': SoftmaxNACCell,
    'SoftmaxNALU': SoftmaxNALUCell,

    'IndependentNAC': IndependentNACCell,
    'IndependentNALU': IndependentNALUCell,

    'HardSoftmaxNAC': HardSoftmaxNACCell,
    'HardSoftmaxNALU': HardSoftmaxNALUCell,

    'GradientBanditNAC': GradientBanditNACCell,
    'GradientBanditNALU': GradientBanditNALUCell,

    'RegualizedLinearNAC': RegualizedLinearNACCell,
    'RegualizedLinearNALU': RegualizedLinearNALUCell,

    'ReRegualizedLinearNAC': ReRegualizedLinearNACCell,
    'ReRegualizedLinearMNAC': ReRegualizedLinearMNACCell,
    'ReRegualizedLinearNALU': ReRegualizedLinearNALUCell,
}

class GeneralizedLayer(ExtendedTorchModule):
    """Abstracts all layers, both basic, NAC and NALU

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
        unit_name: name of the unit (e.g. NAC, Sigmoid, Tanh)
    """
    UNIT_NAMES = set(unit_name_to_layer_class.keys()) | BasicLayer.ACTIVATIONS

    def __init__(self, in_features, out_features, unit_name, writer=None, name=None, **kwags):
        super().__init__('layer', name=name, writer=writer, **kwags)
        self.in_features = in_features
        self.out_features = out_features
        self.unit_name = unit_name

        if unit_name in unit_name_to_layer_class:
            Layer = unit_name_to_layer_class[unit_name]
            self.layer = Layer(in_features, out_features,
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
    UNIT_NAMES = set(unit_name_to_cell_class.keys()) | {'GRU', 'LSTM', 'RNN-tanh', 'RNN-ReLU', 'RNN-linear'}

    def __init__(self, input_size, hidden_size, unit_name, writer=None, **kwags):
        super().__init__('cell', writer=writer, **kwags)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.unit_name = unit_name

        if unit_name in unit_name_to_cell_class:
            Cell = unit_name_to_cell_class[unit_name]
            self.cell = Cell(input_size, hidden_size,
                             writer=self.writer,
                             **kwags)
        elif unit_name == 'none':
            self.cell = PassThoughCell(input_size, hidden_size,
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
        elif unit_name == 'RNN-linear':
            self.cell = BasicCell(input_size, hidden_size,
                                  activation='linear',
                                  writer=self.writer,
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
