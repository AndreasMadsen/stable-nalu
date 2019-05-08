
import os
import shutil
import os.path as path
import torch
from tensorboardX import SummaryWriter as SummaryWriterRaw

THIS_DIR = path.dirname(path.realpath(__file__))

if 'TENSORBOARD_DIR' in os.environ:
    TENSORBOARD_DIR = os.environ['TENSORBOARD_DIR']
else:
    TENSORBOARD_DIR = path.join(THIS_DIR, '../../tensorboard')

class SummaryWriterNamespaceNoLoggingScope:
    def __init__(self, writer):
        self._writer = writer

    def __enter__(self):
        self._writer._logging_enabled = False

    def __exit__(self, type, value, traceback):
        self._writer._logging_enabled = True
        return False

class SummaryWriterNamespace:
    def __init__(self, namespace='', epoch_interval=1, root=None, parent=None):
        self._namespace = namespace
        self._epoch_interval = epoch_interval
        self._parent = parent
        self._logging_enabled = True

        if root is None:
            self._root = self
        else:
            self._root = root

    def get_iteration(self):
        return self._root.get_iteration()

    def is_log_iteration(self):
        return self._root.get_iteration() % self._epoch_interval == 0

    def is_logging_enabled(self):
        writer = self
        while writer is not None:
            if writer._logging_enabled:
                writer = writer._parent
            else:
                return False
        return True

    def add_scalar(self, name, value):
        if self.is_log_iteration() and self.is_logging_enabled():
            self._root.writer.add_scalar(f'{self._namespace}/{name}', value, self.get_iteration())

    def add_summary(self, name, tensor):
        if self.is_log_iteration() and self.is_logging_enabled():
            self._root.writer.add_scalar(f'{self._namespace}/{name}/mean', torch.mean(tensor), self.get_iteration())
            self._root.writer.add_scalar(f'{self._namespace}/{name}/var', torch.var(tensor), self.get_iteration())

    def add_tensor(self, name, matrix):
        if self.is_log_iteration() and self.is_logging_enabled():
            data = matrix.detach().cpu().numpy()
            self._root.writer.add_text(f'{self._namespace}/{name}', f'<pre>{data}</pre>', self.get_iteration())

    def add_histogram(self, name, tensor):
        if torch.isnan(tensor).any():
            print(f'nan detected in {self._namespace}/{name}')
            tensor = torch.where(torch.isnan(tensor), torch.tensor(0, dtype=tensor.dtype), tensor)
            raise ValueError('nan detected')

        if self.is_log_iteration() and self.is_logging_enabled():
            self._root.writer.add_histogram(f'{self._namespace}/{name}', tensor, self.get_iteration())

    def print(self, name, tensor):
        if self.is_log_iteration() and self.is_logging_enabled():
            print(f'{self._namespace}/{name}:')
            print(tensor)

    def namespace(self, name):
        return SummaryWriterNamespace(
            namespace=f'{self._namespace}/{name}',
            epoch_interval=self._epoch_interval,
            root=self._root,
            parent=self,
        )

    def every(self, epoch_interval):
        return SummaryWriterNamespace(
            namespace=self._namespace,
            epoch_interval=epoch_interval,
            root=self._root,
            parent=self,
        )

    def no_logging(self):
        return SummaryWriterNamespaceNoLoggingScope(self)

class SummaryWriter(SummaryWriterNamespace):
    def __init__(self, name, remove_existing_data=False, **kwargs):
        super().__init__()
        self._iteration = 0

        log_dir = path.join(TENSORBOARD_DIR, name)
        if path.exists(log_dir):
            shutil.rmtree(log_dir)

        self.writer = SummaryWriterRaw(log_dir=log_dir, **kwargs)

    def set_iteration(self, iteration):
        self._iteration = iteration

    def get_iteration(self):
        return self._iteration

    def close(self):
        self.writer.close()

    def __del__(self):
        self.close()

class DummySummaryWriter():
    def __init__(self, **kwargs):
        self._logging_enabled = False
        pass

    def add_scalar(self, name, value):
        pass

    def add_summary(self, name, tensor):
        pass

    def add_histogram(self, name, tensor):
        pass

    def add_tensor(self, name, tensor):
        pass

    def print(self, name, tensor):
        pass

    def namespace(self, name):
        return self

    def every(self, epoch_interval):
        return self

    def no_logging(self):
        return SummaryWriterNamespaceNoLoggingScope(self)
