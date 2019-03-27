
import numpy as np
import torch

from stable_nalu.functional import NACWeight

def test_nac_weight_calculates_backward_correctly():
    w_hat = torch.randn(100, 2, requires_grad=True, dtype=torch.float64)
    m_hat = torch.randn(100, 2, requires_grad=True, dtype=torch.float64)

    torch.autograd.gradcheck(
        lambda w_hat, m_hat: 2 * NACWeight.apply(w_hat * 2, m_hat * 2),
        [w_hat, m_hat]
    )
