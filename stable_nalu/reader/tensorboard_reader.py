
import os
import os.path as path
import tensorflow as tf

def _listdir_filter_hidden_files(dirpath):
    files = os.listdir(dirpath)
    files = filter(lambda filename: filename[0] != '.', files)
    return list(files)

class TensorboardReader:
    """Reads the final values (before reaching NaN) from a directory of
    of results
    """

    def __init__(self, dirpath, auto_open=True):
        self._sourcedir = dirpath
        self._directories = _listdir_filter_hidden_files(dirpath)
        self._auto_open = auto_open

    def __iter__(self):
        """Return the last non-nan result from each directory.

        The format is (dirname, losses, last_global_step)
        """
        for subdir in self._directories:
            logfiles = _listdir_filter_hidden_files(path.join(self._sourcedir, subdir))
            if len(logfiles) != 1:
                raise Exception(f'wrong number of logfile was found in {subdir}')

            filename = path.join(self._sourcedir, subdir, logfiles[0])

            if self._auto_open:
                reader = tf.train.summary_iterator(filename)
                yield (subdir, filename, reader)
            else:
                yield (subdir, filename, None)

    def __len__(self):
        return len(self._directories)
