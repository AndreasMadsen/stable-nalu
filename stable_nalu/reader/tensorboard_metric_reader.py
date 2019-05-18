
import re
import ast
import pandas
import collections
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

def _csv_format_column_name(column_name):
    return column_name.replace('/', '.')

def _everything_default_matcher(tag):
    return True

class TensorboardMetricReader:
    def __init__(self, dirname,
                 metric_matcher=_everything_default_matcher,
                 step_start=0,
                 processes=None,
                 progress_bar=True):
        self.dirname = dirname
        self.metric_matcher = metric_matcher
        self.step_start = step_start

        self.processes = processes
        self.progress_bar = progress_bar

    def _parse_tensorboard_data(self, inputs):
        (dirname, filename, reader) = inputs

        columns = collections.defaultdict(list)
        columns['name'] = dirname

        missing_sparse_error = False
        sparse_errors_inserted = 0
        sparse_error_first_collected = True
        sparse_error_collected_at = None
        sparse_error_max = 0
        sparse_error_sum = 0
        sparse_error_count = 0

        for e in tf.train.summary_iterator(filename):
            step = e.step - self.step_start

            for v in e.summary.value:

                if self.metric_matcher(v.tag):
                    columns[v.tag].append(v.simple_value)

                    # Syncronize the step count with the loss metrics
                    if len(columns['step']) != len(columns[v.tag]):
                        columns['step'].append(step)

                    # Syncronize with the next sampled sparse.error, this should
                    # be from the same step.
                    if sparse_errors_inserted != len(columns[v.tag]):
                        missing_sparse_error = True

                elif v.tag.endswith('W/text_summary'):
                    W = _parse_numpy_str(v.tensor.string_val[0][5:-6].decode('ascii'))
                    W_error = np.minimum(np.abs(W), np.abs(1 - np.abs(W)))

                    # Step changed, update sparse error
                    if step != sparse_error_collected_at and not sparse_error_first_collected:
                        if missing_sparse_error:
                            columns['sparse.error.max'].append(sparse_error_max)
                            columns['sparse.error.sum'].append(sparse_error_sum)
                            columns['sparse.error.count'].append(sparse_error_count)
                            missing_sparse_error = False
                            sparse_errors_inserted += 1

                        sparse_error_max = np.max(W_error)
                        sparse_error_sum = np.sum(W_error)
                        sparse_error_count = W_error.size
                    else:
                        sparse_error_max = max(sparse_error_max, np.max(W_error))
                        sparse_error_sum += np.sum(W_error)
                        sparse_error_count += W_error.size

                    sparse_error_collected_at = step
                    sparse_error_first_collected = False

        if missing_sparse_error:
            columns['sparse.error.max'].append(sparse_error_max)
            columns['sparse.error.sum'].append(sparse_error_sum)
            columns['sparse.error.count'].append(sparse_error_count)

        return columns

    def __iter__(self):
        reader = TensorboardReader(self.dirname, auto_open=False)
        with tqdm(total=len(reader), disable=not self.progress_bar) as pbar, \
             multiprocessing.Pool(self.processes) as pool:

            columns_order = None
            #for data in pool.imap_unordered(self._parse_tensorboard_data, reader):
            for data in map(self._parse_tensorboard_data, reader):
                pbar.update()
                df = pandas.DataFrame(data)

                # Ensure the columns are always order the same
                if columns_order is None:
                    columns_order = df.columns.tolist()
                else:
                    df = df[columns_order]

                df.rename(_csv_format_column_name, axis='columns', inplace=True)
                yield df