
import os.path as path
import torch
from tensorboardX import SummaryWriter as SummaryWriterRaw

THIS_DIR = path.dirname(path.realpath(__file__))
TENSORBOARD_DIR = path.join(THIS_DIR, '../../tensorboard')

class SummaryWriterNamespace:
    def __init__(self, namespace='', epoch_interval=1, root=None):
        self._namespace = namespace
        self._epoch_interval = epoch_interval

        if root is None:
            self._root = self
        else:
            self._root = root

    def set_iteration(self, iteration):
        self._root.set_iteration(iteration)

    def get_iteration(self):
        return self._root.get_iteration()

    def _is_log_iteration(self):
        return self._root.get_iteration() % self._epoch_interval == 0

    def add_scalar(self, name, value):
        if self._is_log_iteration():
            self._root.writer.add_scalar(f'{self._namespace}/{name}', value, self.get_iteration())

    def add_summary(self, name, tensor):
        if self._is_log_iteration():
            self.add_scalar(f'{name}/mean', torch.mean(tensor))
            self.add_scalar(f'{name}/var', torch.var(tensor))

    def add_histogram(self, name, tensor):
        if torch.isnan(tensor).any():
            print(f'nan detected in {self._namespace}/{name}')
            tensor = torch.where(torch.isnan(tensor), torch.tensor(0, dtype=tensor.dtype), tensor)
            raise ValueError('nan detected')

        if self._is_log_iteration():
            self._root.writer.add_histogram(f'{self._namespace}/{name}', tensor, self.get_iteration())

    def namespace(self, name):
        return SummaryWriterNamespace(
            namespace=f'{self._namespace}/{name}',
            epoch_interval=self._epoch_interval,
            root=self._root,
        )

    def every(self, epoch_interval):
        return SummaryWriterNamespace(
            namespace=self._namespace,
            epoch_interval=epoch_interval,
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

class DummySummaryWriter():
    def __init__(self):
        pass

    def add_scalar(self, name, value):
        pass

    def add_summary(self, name, tensor):
        pass

    def add_histogram(self, name, tensor):
        pass

    def namespace(self, name):
        return self

    def every(self, epoch_interval):
        return self