
import numpy as np
import torch

from stable_nalu.functional.gated_choice import GatedChoiceNormal

def test_gated_choice_calculates_backward_correctly_indpendent():
    g_hat = torch.randn(20, 2, requires_grad=True, dtype=torch.float64)
    a_hat = torch.randn(20, 2, requires_grad=True, dtype=torch.float64)
    m_hat = torch.randn(20, 2, requires_grad=True, dtype=torch.float64)

    torch.autograd.gradcheck(
        lambda g_hat, a_hat, m_hat: torch.sum((2 * GatedChoiceNormal.apply(
            torch.sigmoid(g_hat),
            torch.sigmoid(a_hat),
            torch.tanh(m_hat),
            mode='test'
        ) - 0)**2),
        [g_hat, a_hat, m_hat]
    )

def test_gated_choice_calculates_backward_correctly_dependent():
    g_hat = torch.randn(20, 2, requires_grad=True, dtype=torch.float64)
    w_hat = torch.randn(20, 2, requires_grad=True, dtype=torch.float64)

    torch.autograd.gradcheck(
        lambda g_hat, a_hat, m_hat: torch.sum((2 * GatedChoiceNormal.apply(
            torch.sigmoid(g_hat),
            torch.tanh(m_hat) * torch.sigmoid(w_hat),
            torch.tanh(m_hat) * torch.sigmoid(w_hat),
            mode='test'
        ) - 0)**2),
        [g_hat, w_hat, m_hat]
    )

test_gated_choice_calculates_backward_correctly_indpendent()
test_gated_choice_calculates_backward_correctly_dependent()
