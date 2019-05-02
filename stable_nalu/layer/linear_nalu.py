
from .linear_nac import LinearNACLayer
from ._abstract_nalu import AbstractNALULayer
from ._abstract_recurrent_cell import AbstractRecurrentCell

class LinearNALULayer(AbstractNALULayer):
    """Implements the NALU (Neural Arithmetic Logic Unit)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(LinearNACLayer, None, in_features, out_features, **kwargs)

class LinearNALUCell(AbstractRecurrentCell):
    """Implements the NALU (Neural Arithmetic Logic Unit) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(LinearNALULayer, None, input_size, hidden_size, **kwargs)

