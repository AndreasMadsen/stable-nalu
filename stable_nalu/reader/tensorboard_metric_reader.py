
import re
import ast
import multiprocessing

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from .tensorboard_reader import TensorboardReader

def _parse_numpy_str(array_string):
    pattern = r'''# Match (mandatory) whitespace between...
                (?<=\]) # ] and
                \s+
                (?= \[) # [, or
                |
                (?<=[^\[\]\s])
                \s+
                (?= [^\[\]\s]) # two non-bracket non-whitespace characters
            '''

    # Replace such whitespace with a comma
    fixed_string = re.sub(pattern, ',', array_string, flags=re.VERBOSE)
    return np.array(ast.literal_eval(fixed_string))

def _parse_tensorboard_data(inputs):
    (dirname, filename, reader) = inputs
    data = []

    is_nalu = False
    interpolation_value = None
    extrapolation_value = None
    sparse_error_max_values = []
    sparse_error_sum_values = []
    sparse_error_count_values = []

    for e in tf.train.summary_iterator(filename):
        for v in e.summary.value:
            if v.tag == 'loss/valid/interpolation':
                interpolation_value = v.simple_value
            elif v.tag == 'loss/valid/extrapolation':
                extrapolation_value = v.simple_value
            elif v.tag.endswith('W/text_summary'):
                if '/nalu/' in v.tag:
                    is_nalu = True

                W = _parse_numpy_str(v.tensor.string_val[0][5:-6].decode('ascii'))
                W_error = np.minimum(np.abs(W), np.abs(1 - np.abs(W)))

                sparse_error_sum_values.append(np.sum(W_error))
                sparse_error_count_values.append(W_error.size)
                sparse_error_max_values.append(np.max(W_error))

        if e.step % 1000 == 0:
            if (interpolation_value is not None and extrapolation_value is not None and (
                len(sparse_error_count_values) == (4 if is_nalu else 2)
            )):
                data.append((
                    e.step,
                    dirname,
                    interpolation_value,
                    extrapolation_value,
                    np.sum(sparse_error_sum_values) / np.sum(sparse_error_count_values),
                    np.max(sparse_error_max_values),
                ))

                is_nalu = False
                interpolation_value = None
                extrapolation_value = None
                sparse_error_max_values = []
                sparse_error_sum_values = []
                sparse_error_count_values = []

    return data

class TensorboardMetricReader:
    def __init__(self, dirname, processes=None, progress_bar=True):
        self.dirname = dirname
        self.processes = processes
        self.progress_bar = progress_bar

    COLUMN_NAMES=('step', 'name', 'interpolation', 'extrapolation', 'sparse.error.mean', 'sparse.error.max')

    def __iter__(self):
        reader = TensorboardReader(self.dirname, auto_open=False)
        with tqdm(total=len(reader), disable=not self.progress_bar) as pbar, \
             multiprocessing.Pool(self.processes) as pool:
            for data in pool.imap_unordered(_parse_tensorboard_data, reader):
                pbar.update()
                yield from data
