
from nose.tools import *

import scipy.stats
import torch
import numpy as np

from stable_nalu.dataset import SimpleFunctionRecurrentDataset

def test_batch_shape():
    dataset = SimpleFunctionRecurrentDataset(
        operation='add', seed=0
    )
    dataset_test = iter(dataset.fork(seq_length=16).dataloader(batch_size=128))
    x_batch, t_batch = next(dataset_test)
    assert_equal(x_batch.size(), (128, 16, 10))
    assert_equal(t_batch.size(), (128, 1))

def test_observation_shape():
    dataset = SimpleFunctionRecurrentDataset(
        operation='add', seed=0
    )
    x_batch, t_batch = dataset.fork(seq_length=16)[0]
    assert_equal(x_batch.size(), (16, 10))
    assert_equal(t_batch.size(), (1, ))
