
from nose.tools import *

import numpy as np
import torch

from stable_nalu.layer import BasicLayer

def test_basic_layer_linear():
    x_np = np.random.RandomState(1).randn(64, 100).astype(np.float32)
    x_tensor = torch.tensor(x_np)

    torch.manual_seed(1)
    layer = BasicLayer(100, 2, activation='linear')
    layer.reset_parameters()
    y_tensor = layer(x_tensor)

    w_np = layer.weight.detach().numpy()
    y_np = np.dot(x_np, np.transpose(w_np))

    np.testing.assert_almost_equal(y_np, y_tensor.detach().numpy())
    assert_equal(y_tensor.shape, (64, 2))
