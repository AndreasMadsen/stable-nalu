
import torch

class RegualizerNMUZ:
    def __init__(self, zero=False):
        self.zero = zero
        self.stored_inputs = []

    def __call__(self, W):
        if self.zero:
            return 0

        x_mean = torch.mean(
            torch.cat(self.stored_inputs, dim=0),
            dim=0, keepdim=True
        )
        return torch.mean((1 - W) * (1 - x_mean)**2)

    def append_input(self, x):
        if self.zero:
            return
        self.stored_inputs.append(x)

    def reset(self):
        if self.zero:
            return
        self.stored_inputs = []
