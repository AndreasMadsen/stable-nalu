
import os
import csv
import sys
import argparse

import stable_nalu

# Parse arguments
parser = argparse.ArgumentParser(description='Export results from simple function task')
parser.add_argument('--tensorboard-dir',
                    action='store',
                    type=str,
                    help='Specify the directory for which the data is stored')
parser.add_argument('--csv-out',
                    action='store',
                    type=str,
                    help='Specify the file for which the csv data is stored at')
args = parser.parse_args()

# Set threads
if 'LSB_DJOB_NUMPROC' in os.environ:
    allowed_processes = int(os.environ['LSB_DJOB_NUMPROC'])
else:
    allowed_processes = None

def matcher(tag):
    return (
        (
            tag.startswith('metric/train') or
            tag.startswith('metric/valid') or
            tag.startswith('metric/test/extrapolation/')
        ) and tag.endswith('/mse')
    )

reader = stable_nalu.reader.TensorboardMetricReader(
    args.tensorboard_dir,
    metric_matcher=matcher,
    recursive_weight=True,
    step_start=1,
    processes=allowed_processes
)

with open(args.csv_out, 'w') as csv_fp:
    for index, df in enumerate(reader):
        df.to_csv(csv_fp, header = (index == 0), index=False)
        csv_fp.flush()
