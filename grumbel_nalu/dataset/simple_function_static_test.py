
from nose.tools import *

import scipy.stats
import torch
import numpy as np

from grumbel_nalu.dataset import SimpleFunctionStaticDataset

def test_solveable_by_linear_algebra():
    dataset = SimpleFunctionStaticDataset('add', seed=0)
    x_batch = []
    t_batch = []
    for i in range(100):
        x, t = dataset[i]
        x_batch.append(x.numpy())
        t_batch.append(t.numpy())

    x_batch_np = np.stack(x_batch)
    t_batch_np = np.stack(t_batch)

    w_merged_np = np.linalg.solve(x_batch_np, t_batch_np.ravel())
    w_merged_np_int = np.round(w_merged_np, 0).astype('int8')

    # W is whole numbers
    np.testing.assert_almost_equal(
        w_merged_np - np.around(w_merged_np, 0),
        np.zeros(100),
        decimal=5
    )
    # W is either 0, 1, 2
    # NOTE: a different seed might not result in an overlap, thus {2} might
    # not be present.
    assert_equal(
        set(np.round(w_merged_np, 0).astype('int8').tolist()),
        {0, 1, 2}
    )

    # Compute a, b range parameters
    # For seed=0, the b subset, is a subset of the a subset, which is assumed
    # by the following algorithm.
    a_start = None
    a_end = None
    b_start = None
    b_end = None

    previuse_w_value = 0
    for w_index, w_value in enumerate(w_merged_np_int.tolist()):
        if w_value == 1 and previuse_w_value == 0:
            a_start = w_index
        elif w_value == 0 and previuse_w_value == 1:
            a_end = w_index
        elif w_value == 2 and previuse_w_value == 1:
            b_start = w_index
        elif w_value == 1 and previuse_w_value == 2:
            b_end = w_index

        previuse_w_value = w_value

    # Compare a and b range parameters
    assert_equal(a_start, dataset.a_start)
    assert_equal(a_end, dataset.a_end)
    assert_equal(b_start, dataset.b_start)
    assert_equal(b_end, dataset.b_end)

def test_input_range():
    dataset = SimpleFunctionStaticDataset('add', input_size=10000, seed=0)
    x, t = dataset[0]
    _, p = scipy.stats.kstest(
        x,
        scipy.stats.uniform(loc=-5, scale=10).cdf
    )
    assert p > 0.5

def test_output_shape():
    dataset = SimpleFunctionStaticDataset('add', seed=0)
    x, t = dataset[0]
    assert_equal(x.shape, (100, ))
    # Note, t.shape should be a 1-long vector, not a scalar. Otherwise
    # the loss function gets confused about what the observation dimention
    # is.
    assert_equal(t.shape, (1, ))
