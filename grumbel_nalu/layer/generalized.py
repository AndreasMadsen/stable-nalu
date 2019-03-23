
import torch

from .nac import NACLayer
from .nalu import NALULayer
from .basic import BasicLayer
from ..writer import DummyWriter

class GeneralizedLayer(torch.nn.Module):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
        unit_name: name of the unit (e.g. NAC, Sigmoid, Tanh)
    """

    def __init__(self, in_features, out_features, unit_name,
                 writer=DummyWriter(), **kwags):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.unit_name = unit_name

        if unit_name == 'NAC':
            self.layer = NACLayer(in_features, out_features,
                                  writer=writer.namespace('nac'),
                                  **kwags)
        elif unit_name == 'NALU':
            self.layer = NALULayer(in_features, out_features,
                                   writer=writer.namespace('nalu'),
                                   **kwags)
        else:
            self.layer = BasicLayer(in_features, out_features,
                                    activation=unit_name,
                                    writer=writer.namespace(f'basic_{unit_name}'),
                                    **kwags)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, input):
        return self.layer.forward(input)

    def extra_repr(self):
        return 'in_features={}, out_features={}, unit_name={}'.format(
            self.in_features, self.out_features, self.unit_name
        )
