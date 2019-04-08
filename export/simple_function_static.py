
from tqdm import tqdm
import torch
import pandas as pd
import stable_nalu
import argparse

pd.options.display.max_rows = 1000

# Parse arguments
parser = argparse.ArgumentParser(description='Export results from simple function task')
parser.add_argument('--dir',
                    action='store',
                    default='tensorboard/simple_function_static',
                    type=str,
                    help='Specify the directory for which the data is stored')
args = parser.parse_args()

"""
data = []
for dirname, losses, last_step in tqdm(stable_nalu.reader.TensorboardFinalReader(args.dir)):
    model, operation, seed = dirname.split('_')
    row = {
        'model': model,
        'operation': operation,
        'seed': seed,
        **losses,
        'step': last_step
    }
    data.append(row)

data_df = pd.DataFrame.from_records(
    data,
    columns=[
        'model', 'operation', 'seed', 'step',
        'loss/train', 'loss/valid/interpolation', 'loss/valid/extrapolation'
    ]
)
data_df.to_pickle('debug_simple_function_static.pkl')
"""
data_df = pd.read_pickle('debug_simple_function_static.pkl')

# Compute baselines
baselines = []
for operation in tqdm(['add', 'sub', 'mul', 'div', 'squared', 'root']):
    interpolation_error = 0
    extrapolation_error = 0
    for seed in range(10):
        dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
            operation=operation,
            seed=seed
        )
        interpolation_error += dataset.fork(input_range=1).baseline_error(batch_size=128)
        extrapolation_error += dataset.fork(input_range=5).baseline_error(batch_size=128)

    baselines.append({
        'operation': operation,
        'baseline/interpolation': interpolation_error / 10,
        'baseline/extrapolation': extrapolation_error / 10
    })

baselines_df = pd.DataFrame.from_records(baselines)

df = pd.merge(data_df, baselines_df, how='left', on='operation')
df['loss/norm/train'] = df['loss/train'] / df['baseline/interpolation']
df['loss/norm/valid/interpolation'] = df['loss/valid/interpolation'] / df['baseline/interpolation']
df['loss/norm/valid/extrapolation'] = df['loss/valid/extrapolation'] / df['baseline/extrapolation']

del df['loss/norm/train']
del df['loss/train']
del df['loss/valid/interpolation']
del df['baseline/interpolation']
del df['loss/valid/extrapolation']
del df['baseline/extrapolation']

df['succes/interpolation'] = df['loss/norm/valid/interpolation'] < 0.001
df['succes/extrapolation'] = df['loss/norm/valid/extrapolation'] < 0.001
del df['loss/norm/valid/interpolation']
del df['loss/norm/valid/extrapolation']

agg_df = df.groupby(['model', 'operation']).agg({
    'succes/interpolation': 'mean',
    'succes/extrapolation': 'mean',
    'seed': 'count'
})
print(agg_df)
