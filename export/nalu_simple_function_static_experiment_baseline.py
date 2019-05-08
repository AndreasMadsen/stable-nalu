
import ast
import re
import sys
import argparse

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pandas as pd

import stable_nalu

pd.options.display.max_rows = 1000

# Parse arguments
parser = argparse.ArgumentParser(description='Export results from simple function task')
parser.add_argument('--dir',
                    action='store',
                    default='tensorboard/nalu_simple_function_static_experiment_baseline',
                    type=str,
                    help='Specify the directory for which the data is stored')
args = parser.parse_args()

def parse_numpy_str(array_string):
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

data = []

for dirname, reader in tqdm(stable_nalu.reader.TensorboardReader(args.dir)):
    is_nalu = False
    interpolation_value = None
    extrapolation_value = None
    sparse_error_max_values = []
    sparse_error_sum_values = []
    sparse_error_count_values = []

    for e in reader:
        for v in e.summary.value:
            if v.tag == 'loss/valid/interpolation':
                interpolation_value = v.simple_value
            elif v.tag == 'loss/valid/extrapolation':
                extrapolation_value = v.simple_value
            elif v.tag.endswith('W/text_summary'):
                if '/nalu/' in v.tag:
                    is_nalu = True

                W = parse_numpy_str(v.tensor.string_val[0][5:-6].decode('ascii'))
                W_error = np.minimum(np.abs(W), np.abs(1 - np.abs(W)))

                sparse_error_sum_values.append(np.sum(W_error))
                sparse_error_count_values.append(W_error.size)
                sparse_error_max_values.append(np.max(W_error))

        if e.step % 1000 == 0:
            if (interpolation_value is not None and extrapolation_value is not None and (
                len(sparse_error_count_values) == (4 if is_nalu else 2)
            )):
                data.append([
                    e.step,
                    dirname,
                    interpolation_value,
                    extrapolation_value,
                    np.sum(sparse_error_sum_values) / np.sum(sparse_error_count_values),
                    np.max(sparse_error_max_values),
                ])

                is_nalu = False
                interpolation_value = None
                extrapolation_value = None
                sparse_error_max_values = []
                sparse_error_sum_values = []
                sparse_error_count_values = []


df = pd.DataFrame.from_records(data, columns=('step', 'name', 'interpolation', 'extrapolation', 'sparse.error.mean', 'sparse.error.max'))
df.to_csv('./data/nalu_simple_function_static_experiment_baseline.csv', index=False)
