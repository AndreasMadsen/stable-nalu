
from .softmax_nac import SoftmaxNACLayer
from ._abstract_nalu import AbstractNALULayer
from ._abstract_recurrent_cell import AbstractRecurrentCell

class SoftmaxNALULayer(AbstractNALULayer):
    """Implements the NALU (Neural Arithmetic Logic Unit)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(SoftmaxNACLayer, None, in_features, out_features, **kwargs)

class SoftmaxNALUCell(AbstractRecurrentCell):
    """Implements the NALU (Neural Arithmetic Logic Unit) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(SoftmaxNALULayer, input_size, hidden_size, **kwargs)
