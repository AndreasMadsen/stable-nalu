
import torch
from ..writer import DummySummaryWriter

class ExtendedTorchModule(torch.nn.Module):
    def __init__(self, default_layer_name, *args, writer=None, layer_name=None, **kwargs):
        super().__init__()
        if writer is None:
            writer = DummySummaryWriter()

        self.writer = writer.namespace(default_layer_name if layer_name is None else layer_name)

    def apply_recursive(self, method_name, *args, **kwargs):
        if hasattr(self, method_name):
            getattr(self, method_name)(*args, **kwargs)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.apply_recursive(method_name, *args, **kwargs)

    def set_parameter(self, name, value, deep=True):
        parameter = getattr(self, name, None)
        if isinstance(parameter, torch.nn.Parameter):
            parameter.fill_(value)

        if deep:
            self.apply_recursive('set_parameter', name, value, deep=False)
