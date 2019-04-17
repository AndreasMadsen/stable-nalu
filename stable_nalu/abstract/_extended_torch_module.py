
import torch
from ..writer import DummySummaryWriter

class ExtendedTorchModule(torch.nn.Module):
    def __init__(self, default_name, *args, writer=None, name=None, **kwargs):
        super().__init__()
        if writer is None:
            writer = DummySummaryWriter()

        self.writer = writer.namespace(default_name if name is None else name)

    def set_parameter(self, name, value):
        parameter = getattr(self, name, None)
        if isinstance(parameter, torch.nn.Parameter):
            parameter.fill_(value)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.set_parameter(name, value)

    def regualizer(self):
        regualizer_sum = 0

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                regualizer_sum += module.regualizer()

        return regualizer_sum

    def optimize(self, loss):
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.optimize(loss)

    def log_gradients(self):
        for name, parameter in self.named_parameters(recurse=False):
            if parameter.requires_grad:
                gradient, *_ = parameter.grad.data
                self.writer.add_summary(f'{name}/grad', gradient)
                self.writer.add_histogram(f'{name}/grad', gradient)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.log_gradients()
