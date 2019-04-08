
import os
import math
import os.path as path
import tensorflow as tf
import multiprocessing

def _listdir_filter_hidden_files(dirpath):
    files = os.listdir(dirpath)
    files = filter(lambda filename: filename[0] != '.', files)
    return list(files)

def _get_losses_from_file(filepath):
    losses = {}
    last_step = 0
    for e in tf.train.summary_iterator(filepath):
        last_step = e.step
        for v in e.summary.value:
            if v.tag[0:5] == 'loss/':
                if not math.isfinite(v.simple_value):
                    return (losses, last_step)
                losses[v.tag] = v.simple_value
    return (losses, last_step)


class TensorboardFinalReader:
    """Reads the final values (before reaching NaN) from a directory of
    of results
    """

    def __init__(self, dirpath):
        self._sourcedir = dirpath
        self._directories = _listdir_filter_hidden_files(dirpath)

    def __iter__(self):
        """Return the last non-nan result from each directory.

        The format is (dirname, losses, last_global_step)
        """
        for subdir in self._directories:
            logfiles = _listdir_filter_hidden_files(path.join(self._sourcedir, subdir))
            if len(logfiles) > 1:
                raise Exception(f'more than one logfile was found in {subdir}')

            losses = _get_losses_from_file(path.join(self._sourcedir, subdir, logfiles[0]))
            yield (subdir, *losses)

    def __len__(self):
        return len(self._directories)
