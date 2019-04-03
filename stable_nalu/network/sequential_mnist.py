
import torch
from ..layer import GeneralizedLayer, GeneralizedCell
from ..writer import DummyWriter

# Copied from https://github.com/pytorch/examples/blob/master/mnist/main.py, just added a
# reset_parameters method and changed log_softmax to softmax.

class _Image2LabelCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.softmax(x, dim=1)  # do we want a softmax?

class SequentialMnistNetwork(torch.nn.Module):
    def __init__(self, unit_name, output_size,
                 writer=DummyWriter(), **kwags):
        super().__init__()
        self.unit_name = unit_name
        self.output_size = output_size

        # Since for the 'mul' problem, the zero_state should be 1, and for the
        # 'add' problem it should be 0. The zero_states are allowed to be
        # # optimized.
        if unit_name == 'LSTM':
            self.zero_state = torch.nn.ParameterDict({
                'h_t0': torch.nn.Parameter(torch.Tensor(self.output_size)),
                'c_t0': torch.nn.Parameter(torch.Tensor(self.output_size))
            })
        else:
            self.zero_state = torch.nn.Parameter(torch.Tensor(self.output_size))

        self._image2label = _Image2LabelCNN()
        self.recurent_cell = GeneralizedCell(10, self.output_size,
                                             unit_name,
                                             writer=writer.namespace('recurrent_layer'),
                                             **kwags)

    def reset_parameters(self):
        if self.unit_name == 'LSTM':
            for zero_state in self.zero_state.values():
                torch.nn.init.zeros_(zero_state)
        else:
            torch.nn.init.zeros_(self.zero_state)

        self._image2label.reset_parameters()
        self.recurent_cell.reset_parameters()

    def forward(self, x):
        """Performs recurrent iterations over the input.

        Arguments:
            input: Expected to have the shape [obs, time, channels=1, width, height]
        """
        # Perform recurrent iterations over the input
        if self.unit_name == 'LSTM':
            h_tm1 = tuple(zero_state.repeat(x.size(0), 1) for zero_state in self.zero_state.values())
        else:
            h_tm1 = self.zero_state.repeat(x.size(0), 1)

        for t in range(x.size(1)):
            x_t = x[:, t]
            l_t = self._image2label(x_t)
            h_t = self.recurent_cell.forward(l_t, h_tm1)
            h_tm1 = h_t

        # Grap the final hidden output and use as the output from the recurrent layer
        z_1 = h_t[0] if self.unit_name == 'LSTM' else h_t
        return z_1

    def extra_repr(self):
        return 'unit_name={}, output_size={}'.format(
            self.unit_name, self.output_size
        )
