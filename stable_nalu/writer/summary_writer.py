
import os.path as path
import torch
from tensorboardX import SummaryWriter as SummaryWriterRaw

THIS_DIR = path.dirname(path.realpath(__file__))
TENSORBOARD_DIR = path.join(THIS_DIR, '../../tensorboard')

class SummaryWriterNamespace:
    def __init__(self, namespace='', root=None):
        self._namespace = namespace

        if root is None:
            self._root = self
        else:
            self._root = root

    def set_iteration(self, iteration):
        self._root.set_iteration(iteration)

    def get_iteration(self):
        return self._root.get_iteration()

    def add_scalar(self, name, value):
        self._root.writer.add_scalar(f'{self._namespace}/{name}', value, self.get_iteration())

    def add_summary(self, name, tensor):
        self.add_scalar(f'{name}/mean', torch.mean(tensor))
        self.add_scalar(f'{name}/var', torch.var(tensor))

    def namespace(self, name):
        return SummaryWriterNamespace(
            namespace=f'{self._namespace}/{name}',
            root=self._root,
        )

class SummaryWriter(SummaryWriterNamespace):
    def __init__(self, name, **kwargs):
        super().__init__()
        self._iteration = 0
        self.writer = SummaryWriterRaw(
            log_dir=path.join(TENSORBOARD_DIR, name),
            **kwargs)

    def set_iteration(self, iteration):
        self._iteration = iteration

    def get_iteration(self):
        return self._iteration

    def close(self):
        self.writer.close()

    def __del__(self):
        self.close()

class DummyWriter():
    def __init__(self):
        pass

    def add_scalar(self, name, value):
        pass

    def add_summary(self, name, tensor):
        pass

    def namespace(self, name):
        return self
