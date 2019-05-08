
import os
import math
import os.path as path
import tensorflow as tf
from .tensorboard_reader import TensorboardReader

def _listdir_filter_hidden_files(dirpath):
    files = os.listdir(dirpath)
    files = filter(lambda filename: filename[0] != '.', files)
    return list(files)

def _get_losses_from_file(reader):
    losses = {}
    last_step = 0
    for e in reader:
        last_step = e.step
        for v in e.summary.value:
            if v.tag[0:5] == 'loss/':
                if not math.isfinite(v.simple_value):
                    return (losses, last_step)
                losses[v.tag] = v.simple_value
    return (losses, last_step)


class TensorboardFinalReader(TensorboardReader):
    """Reads the final values (before reaching NaN) from a directory of
    of results
    """

    def __init__(self, dirpath):
        self._reader = TensorboardReader(dirpath)

    def __iter__(self):
        """Return the last non-nan result from each directory.

        The format is (dirname, losses, last_global_step)
        """
        for subdir, reader in self._reader:
            losses = _get_losses_from_reader(reader)
            yield (subdir, *losses)

    def __len__(self):
        return len(self._reader)
