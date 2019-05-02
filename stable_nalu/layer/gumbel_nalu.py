
from .gumbel_nac import GumbelNACLayer
from .gumbel_mnac import GumbelMNACLayer
from ._abstract_nalu import AbstractNALULayer
from ._abstract_recurrent_cell import AbstractRecurrentCell

class GumbelNALULayer(AbstractNALULayer):
    """Implements the Gumbel NALU (Neural Arithmetic Logic Unit)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(GumbelNACLayer, GumbelMNACLayer, in_features, out_features, **kwargs)

class GumbelNALUCell(AbstractRecurrentCell):
    """Implements the Gumbel NALU (Neural Arithmetic Logic Unit) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(GumbelNALULayer, GumbelMNACLayer, input_size, hidden_size, **kwargs)
