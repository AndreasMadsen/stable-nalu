
import torch

def batch_linear(x, W, b=None):
    """Computes y_i = x_i W_i + b_i where i is each observation index.

    This is similar to `torch.nn.functional.linear`, but a version that
    supports a different W for each observation.

    x: has shape [obs, in_dims]
    W: has shape [obs, out_dims, in_dims]
    b: has shape [out_dims]
    """
    if x.size()[1] != W.size()[-1]:
        raise ValueError(
            f'the in_dim of x ({x.size()[1]}) does not match in_dim of W ({W.size()[-1]})')

    if x.size()[0] != W.size()[0]:
        raise ValueError(
            f'the obs of x ({x.size()[0]}) does not match obs of W ({W.size()[0]})')

    obs = x.size()[0]
    in_dims = x.size()[1]
    out_dims = W.size()[1]

    x = x.view(obs, 1, in_dims)
    W = W.transpose(-2, -1)

    if b is None:
        return torch.bmm(x, W).view(obs, out_dims)
    else:
        b = b.view(1, 1, out_dims)
        return torch.baddbmm(1, b, 1, x, W).view(obs, out_dims)
