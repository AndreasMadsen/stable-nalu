
import torch
from ..writer import DummySummaryWriter

class ExtendedTorchModule(torch.nn.Module):
    def __init__(self, default_layer_name, *args, writer=None, layer_name=None, **kwargs):
        super().__init__()
        if writer is None:
            writer = DummySummaryWriter()

        self.writer = writer.namespace(default_layer_name if layer_name is None else layer_name)

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
