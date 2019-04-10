
import torch

def sample_gumbel(placeholder, eps=1e-10, reuse=False):
    """Samples Gumbel(0, 1) values into the placeholder"""

    # Uniform sample between [eps, 1)
    if reuse:
        uniform = placeholder
    else:
        uniform = placeholder.uniform_(eps, 1)

    # Inverse transform
    g = -torch.log(-torch.log(uniform))
    return g

def sample_gumbel_softmax(placeholder, logits, tau, **kwargs):
    """Samples values from a gumbel softmax

    Arguments:
        placeholder: A tensor used to specify the device storage
            (cpu or cuda). Note that the content of the placeholder
            will be overwritten.
        logits: log properbilities, you can use log_softmax to
            transform a tensor into log properbilities.
        tau: the temperature used, must be tau \in (0, \infty]. tau < 1
            makes the distribution more categorical. tau > 1 makes
            the distribution more uniform.
    """
    g = sample_gumbel(placeholder, **kwargs)
    return torch.nn.functional.softmax((logits + g) / tau, dim=-1)

def sample_gumbel_max(placeholder, logits, **kwargs):
    """Samples values from a gumbel max

    Arguments:
        placeholder: A tensor used to specify the device storage
            (cpu or cuda). Note that the content of the placeholder
            will be overwritten.
        logits: log properbilities, you can use log_softmax to
            transform a tensor into log properbilities.
    """
    g = sample_gumbel(placeholder, **kwargs)
    indices = torch.argmax(logits + g, dim=-1)

    # Convert indices to a one-hot encoding
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(-1, indices, 1)
    return one_hot
