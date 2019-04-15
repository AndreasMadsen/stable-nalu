
import torch
from ..abstract import ExtendedTorchModule

class AbstractRecurrentCell(ExtendedTorchModule):
    def __init__(self, Op, input_size, hidden_size, writer=None, **kwargs):
        super().__init__('recurrent', writer=writer, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.op = Op(input_size + hidden_size, hidden_size, writer=self.writer, **kwargs)

    def reset_parameters(self):
        self.op.reset_parameters()

    def forward(self, x_t, h_tm1):
        return self.op(torch.cat((x_t, h_tm1), dim=1))

    def extra_repr(self):
        return 'input_size={}, hidden_size={}'.format(
            self.input_size, self.hidden_size
        )
