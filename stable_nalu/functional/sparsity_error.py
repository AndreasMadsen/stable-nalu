
import torch

def sparsity_error(W):
    W_error = torch.min(torch.abs(W), torch.abs(1 - torch.abs(W)))
    return torch.max(W_error)
