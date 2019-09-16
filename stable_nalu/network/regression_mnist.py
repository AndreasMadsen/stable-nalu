
import torch
from ..abstract import ExtendedTorchModule
from ..layer import GeneralizedLayer, GeneralizedCell

# Copied from https://github.com/pytorch/examples/blob/master/mnist/main.py, just added a
# reset_parameters method and changed final layer to have one output.

class RegressionMnistNetwork(ExtendedTorchModule):
    def __init__(self,
                 mnist_digits=[0,1,2,3,4,5,6,7,8,9],
                 softmax_transform=False,
                 mnist_outputs=1, **kwargs):
        super().__init__('cnn', **kwargs)
        self._softmax_transform = softmax_transform

        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*50, 500)
        if self._softmax_transform:
            if mnist_outputs:
                raise ValueError(f"mnist_outputs can't be > 1 with softmax_transform")

            self.fc2 = torch.nn.Linear(500, len(mnist_digits))
            self.register_buffer('fc3', torch.tensor(mnist_digits, dtype=torch.float32).reshape(1, -1))
        else:
            self.fc2 = torch.nn.Linear(500, mnist_outputs)

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

        if self._softmax_transform:
            x = torch.nn.functional.softmax(x, dim=-1)
            x = (x * self.fc3).sum(1, keepdim=True)
        return x
