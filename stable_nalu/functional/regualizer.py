
import torch

class Regualizer:
    def __init__(self, support='nac', type='bias', shape='squared', zero=False, zero_epsilon=0):
        super()
        self.zero_epsilon = 0

        if zero:
            self.fn = self._zero
        else:
            identifier = '_'.join(['', support, type, shape])
            self.fn = getattr(self, identifier)

    def __call__(self, W):
        return self.fn(W)

    def _zero(self, W):
        return 0

    def _mnac_bias_linear(self, W):
        return torch.mean(torch.min(
            torch.abs(W - self.zero_epsilon),
            torch.abs(1 - W)
        ))

    def _mnac_bias_squared(self, W):
        return torch.mean((W - self.zero_epsilon)**2 * (1 - W)**2)

    def _mnac_oob_linear(self, W):
        return torch.mean(torch.relu(
            torch.abs(W - 0.5 - self.zero_epsilon)
            - 0.5 + self.zero_epsilon
        ))

    def _mnac_oob_squared(self, W):
        return torch.mean(torch.relu(
            torch.abs(W - 0.5 - self.zero_epsilon)
            - 0.5 + self.zero_epsilon
        )**2)

    def _nac_bias_linear(self, W):
        W_abs = torch.abs(W)
        return torch.mean(torch.min(
            W_abs,
            torch.abs(1 - W_abs)
        ))

    def _nac_bias_squared(self, W):
        return torch.mean(W**2 * (1 - torch.abs(W))**2)

    def _nac_oob_linear(self, W):
        return torch.mean(torch.relu(torch.abs(W) - 1))

    def _nac_oob_squared(self, W):
        return torch.mean(torch.relu(torch.abs(W) - 1)**2)
