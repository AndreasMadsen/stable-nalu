
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
                 recursive_weight=False,
                 processes=None,
                 progress_bar=True):
        self.dirname = dirname
        self.metric_matcher = metric_matcher
        self.step_start = step_start
        self.recursive_weight = recursive_weight

        self.processes = processes
        self.progress_bar = progress_bar

    def _parse_tensorboard_data(self, inputs):
        (dirname, filename, reader) = inputs

        columns = collections.defaultdict(list)
        columns['name'] = dirname

        current_epoch = None
        current_logged_step = None

        for e in tf.train.summary_iterator(filename):
            step = e.step - self.step_start

            for v in e.summary.value:
                if v.tag == 'epoch':
                    current_epoch = v.simple_value

                elif self.metric_matcher(v.tag):
                    columns[v.tag].append(v.simple_value)
                    current_logged_step = step

                    # Syncronize the step count with the loss metrics
                    if len(columns['step']) != len(columns[v.tag]):
                        columns['step'].append(step)

                    # Syncronize the wall.time with the loss metrics
                    if len(columns['wall.time']) != len(columns[v.tag]):
                        columns['wall.time'].append(e.wall_time)

                    # Syncronize the epoch with the loss metrics
                    if current_epoch is not None and len(columns['epoch']) != len(columns[v.tag]):
                        columns['epoch'].append(current_epoch)

                elif v.tag.endswith('W/text_summary') and current_logged_step == step:
                    if self.recursive_weight:
                        W = _parse_numpy_str(v.tensor.string_val[0][5:-6].decode('ascii'))
                        if len(columns['step']) != len(columns['recursive.weight']):
                            columns['recursive.weight'].append(W[0, -1])

                elif v.tag.endswith('W/sparsity_error') and current_logged_step == step:
                    # Step changed, update sparse error
                    if len(columns['step']) != len(columns['sparse.error.max']):
                        columns['sparse.error.max'].append(v.simple_value)
                    else:
                        columns['sparse.error.max'][-1] = max(
                            columns['sparse.error.max'][-1],
                            v.simple_value
                        )

        if len(columns['sparse.error.max']) == 0:
            columns['sparse.error.max'] = [None] * len(columns['step'])
        if self.recursive_weight:
            if len(columns['recursive.weight']) == 0:
                columns['recursive.weight'] = [None] * len(columns['step'])

        return dirname, columns

    def __iter__(self):
        reader = TensorboardReader(self.dirname, auto_open=False)
        with tqdm(total=len(reader), disable=not self.progress_bar) as pbar, \
             multiprocessing.Pool(self.processes) as pool:

            columns_order = None
            for dirname, data in pool.imap_unordered(self._parse_tensorboard_data, reader):
                pbar.update()

                # Check that some data is present
                # if len(data['step']) == 0:
                #     print(f'missing data in: {dirname}')
                #     continue

                # Fix flushing issue
                for column_name, column_data in data.items():
                    if len(data['step']) - len(column_data) == 1:
                        data[column_name].append(None)

                # Convert to dataframe
                df = pandas.DataFrame(data)
                if len(df) == 0:
                    print(f'Warning: No data for {dirname}')
                    continue

                # Ensure the columns are always order the same
                if columns_order is None:
                    columns_order = df.columns.tolist()
                else:
                    df = df[columns_order]

                df.rename(_csv_format_column_name, axis='columns', inplace=True)
                yield df
