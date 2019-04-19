
import numpy as np
import torch

from stable_nalu.functional import nac_weight

def test_nac_weight_calculates_backward_correctly():
    w_hat = torch.randn(100, 2, requires_grad=True, dtype=torch.float64)
    m_hat = torch.randn(100, 2, requires_grad=True, dtype=torch.float64)

    torch.autograd.gradcheck(
        lambda w_hat, m_hat: torch.sum((2 * nac_weight(w_hat * 2, m_hat * 2) - 0)**2),
        [w_hat, m_hat]
    )
