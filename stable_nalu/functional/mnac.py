
import torch

def mnac(x, W, mode='prod'):
    out_size, in_size = W.size()
    x = x.view(x.size()[0], in_size, 1)
    W = W.t().view(1, in_size, out_size)

    if mode == 'prod':
        return torch.prod(x * W + 1 - W, -2)
    elif mode == 'exp-log':
        return torch.exp(torch.sum(torch.log(x * W + 1 - W), -2))
    else:
        raise ValueError(f'mnac mode "{mode}" is not implemented')
