
import torch
from ..writer import DummySummaryWriter

class ExtendedTorchModule(torch.nn.Module):
    def __init__(self, default_layer_name, *args, writer=None, layer_name=None, **kwargs):
        super().__init__()
        if writer is None:
            writer = DummySummaryWriter()

        self.writer = writer.namespace(default_layer_name if layer_name is None else layer_name)

    def recursive_apply(self, method_name, *args, **kwargs):
        if hasattr(self, method_name):
            getattr(self, method_name)(*args, **kwargs)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.recursive_apply(method_name, *args, **kwargs)
