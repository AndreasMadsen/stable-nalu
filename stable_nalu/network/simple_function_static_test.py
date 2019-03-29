
from nose.tools import *

import torch
import numpy as np

from stable_nalu.dataset import SimpleFunctionStaticDataset
from stable_nalu.network import SimpleFunctionStaticNetwork

def test_linear_solves_add():
    dataset = SimpleFunctionStaticDataset(operation='add', seed=0)
    dataset_test = dataset.fork(input_range=1)

    w_1 = np.zeros((100, 2), dtype=np.float32)
    w_1[dataset.a_start:dataset.a_end, 0] = 1
    w_1[dataset.b_start:dataset.b_end, 1] = 1
    w_2 = np.ones((2, 1), dtype=np.float32)

    network = SimpleFunctionStaticNetwork('linear', input_size=100)
    network.layer_1.layer.weight.data = torch.tensor(np.transpose(w_1))
    network.layer_2.layer.weight.data = torch.tensor(np.transpose(w_2))

    for i, (x, t) in zip(range(100), dataset_test):
        np.testing.assert_almost_equal(
            network(x).detach().numpy(),
            t.numpy(),
            decimal=4
        )
