
import torch
from ..writer import DummyWriter

class AbstractRecurrentCell(torch.nn.Module):
    def __init__(self, Op, input_size, hidden_size, writer=DummyWriter()):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.op = Op(input_size + hidden_size, hidden_size, writer=writer)

    def reset_parameters(self):
        self.op.reset_parameters()

    def forward(self, x_t, h_tm1):
        return self.op(torch.cat((x_t, h_tm1), dim=1))

    def extra_repr(self):
        return 'input_size={}, hidden_size={}'.format(
            self.input_size, self.hidden_size
        )
