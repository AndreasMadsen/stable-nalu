
import math
import torch

from ..abstract import ExtendedTorchModule
from ..functional import sparsity_error
from ._abstract_recurrent_cell import AbstractRecurrentCell

class GradientBanditNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features

        # The learning rate
        self.lr = torch.nn.Parameter(torch.tensor(1e-3, dtype=torch.float32), requires_grad=False)

        # Holds a running mean of the loss
        self.running_mean_beta = torch.nn.Parameter(torch.tensor(0.9, dtype=torch.float32), requires_grad=False)
        self.register_buffer('running_mean_iter', torch.tensor(0, dtype=torch.float32))
        self.register_buffer('running_mean_loss', torch.tensor(0, dtype=torch.float32))

        # Define the target weights. Also, put 0 last such that p1 = p2 = 0
        # corresponds to p3 = 1 => w = 0.
        self.register_buffer('target_weights', torch.tensor([1, -1, 0], dtype=torch.float32))

        # Initialize a tensor, that will be the placeholder for the hard sample
        self.sample = torch.LongTensor(out_features, in_features)

        # We will only two parameters per weight, this is to prevent the redundancy
        # there would otherwise exist. This also makes it much more comparable with
        # NAC.
        self.register_buffer('W_hat', torch.Tensor(out_features, in_features, 3))

        self.register_parameter('bias', None)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.W_hat)
        self.running_mean_iter.fill_(0)
        self.running_mean_loss.fill_(0)

    def optimize(self, loss):
        # Increment the iteration counter
        self.running_mean_iter.add_(1)

        # Update running mean, this is a unbiased exponental average, see Adam()
        self.running_mean_loss.mul_(self.running_mean_beta).add_(1 - self.running_mean_beta, loss)
        running_mean_loss_debias = self.running_mean_loss / (1 - self.running_mean_beta**self.running_mean_iter)

        # Convert W sample to a one-hot encoding
        samples_one_hot = torch.zeros(self.out_features, self.in_features, 3) \
            .scatter_(2, self.sample.view(self.out_features, self.in_features, 1), 1.0)

        # Compute update
        pi = torch.nn.functional.softmax(self.W_hat, dim=-1)
        self.W_hat.addcmul_(self.lr, running_mean_loss_debias - loss, samples_one_hot - pi)

    def forward(self, input, reuse=False):
        pi = torch.nn.functional.softmax(self.W_hat, dim=-1)

        # Compute W
        if self.allow_random:
            if not reuse:
                torch.multinomial(pi.view(-1, 3), 1, True, out=self.sample.view(-1))
            W = self.target_weights[self.sample]
        else:
            W = self.target_weights[torch.argmax(pi, dim=-1)]

        # Compute the linear multiplication as usual
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)

        return torch.nn.functional.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class GradientBanditNACCell(AbstractRecurrentCell):
    """Implements the Gumbel NAC (Gumbel Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(GradientBanditNACLayer, input_size, hidden_size, **kwargs)
