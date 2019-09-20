
import math
import torch

from ..abstract import ExtendedTorchModule

torch.nn.functional.gumbel_softmax

class AbstractNALULayer(ExtendedTorchModule):
    """Implements the NALU (Neural Arithmetic Logic Unit)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, NACOp, MNACOp, in_features, out_features, eps=1e-7,
                 nalu_two_nac=False, nalu_two_gate=False,
                 nalu_bias=False, nalu_mul='normal', nalu_gate='normal',
                 writer=None, name=None, **kwargs):
        super().__init__('nalu', name=name, writer=writer, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.nalu_two_nac = nalu_two_nac
        self.nalu_two_gate = nalu_two_gate
        self.nalu_bias = nalu_bias
        self.nalu_mul = nalu_mul
        self.nalu_gate = nalu_gate

        if nalu_mul == 'mnac' and not nalu_two_nac:
            raise ValueError('nalu_two_nac must be true when mnac is used')

        if nalu_gate == 'gumbel' or nalu_gate == 'obs-gumbel':
            self.tau = torch.tensor(1, dtype=torch.float32)

        if nalu_two_nac and nalu_mul == 'mnac':
            self.nac_add = NACOp(in_features, out_features, writer=self.writer, name='nac_add', **kwargs)
            self.nac_mul = MNACOp(in_features, out_features, writer=self.writer, name='nac_mul', **kwargs)
        elif nalu_two_nac:
            self.nac_add = NACOp(in_features, out_features, writer=self.writer, name='nac_add', **kwargs)
            self.nac_mul = NACOp(in_features, out_features, writer=self.writer, name='nac_mul', **kwargs)
        else:
            self.nac_add = NACOp(in_features, out_features, writer=self.writer, **kwargs)
            self.nac_mul = self._nac_add_reuse

        self.G_add = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if nalu_two_gate:
            self.G_mul = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        if nalu_bias:
            self.bias_add = torch.nn.Parameter(torch.Tensor(out_features))
            if nalu_two_gate:
                self.bias_mul = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_add', None)
            self.register_parameter('bias_mul', None)

        # Don't make this a buffer, as it is not a state that we want to permanently save
        self.stored_gate_add = torch.tensor([0], dtype=torch.float32)
        if nalu_two_gate:
            self.stored_gate_mul = torch.tensor([0], dtype=torch.float32)
        self.stored_input = torch.tensor([0], dtype=torch.float32)

    def _nac_add_reuse(self, x):
        return self.nac_add(x, reuse=True)

    def regualizer(self):
        regualizers = {}

        if self.nalu_gate == 'regualized':
            # NOTE: This is almost identical to sum(g * (1 - g)). Primarily
            # sum(g * (1 - g)) is 4 times larger than sum(g^2 * (1 - g)^2), the curve
            # is also a bit wider. Besides this there is only a very small error.
            regualizers['g'] = torch.sum(self.stored_gate_add**2 * (1 - self.stored_gate_add)**2)
            if self.nalu_two_gate:
                regualizers['g'] += torch.sum(self.stored_gate_mul**2 * (1 - self.stored_gate_mul)**2)

        if self.nalu_gate == 'max-safe':
            regualizers['z'] = torch.mean((1 - self.stored_gate) * torch.relu(1 - self.stored_input))

        # Continue recursion on the regualizer, such that if the NACOp has a regualizer, this is included too.
        return super().regualizer(regualizers)

    def reset_parameters(self):
        self.nac_add.reset_parameters()
        if self.nalu_two_nac:
            self.nac_mul.reset_parameters()

        torch.nn.init.xavier_uniform_(
            self.G_add,
            gain=torch.nn.init.calculate_gain('sigmoid'))
        if self.nalu_two_gate:
            torch.nn.init.xavier_uniform_(
                self.G_mul,
                gain=torch.nn.init.calculate_gain('sigmoid'))

        if self.nalu_bias:
            # consider http://proceedings.mlr.press/v37/jozefowicz15.pdf
            torch.nn.init.constant_(self.bias_add, 0)
            if self.nalu_two_gate:
                torch.nn.init.constant_(self.bias_mul, 0)

    def _compute_gate(self, x, G, bias):
        # g = sigmoid(G x)
        if self.nalu_gate == 'gumbel' or self.nalu_gate == 'obs-gumbel':
            gumbel = 0
            if self.allow_random and self.nalu_gate == 'gumbel':
                gumbel = (-torch.log(1e-8 - torch.log(torch.rand(self.out_features, device=x.device) + 1e-8)))
            elif self.allow_random and self.nalu_gate == 'obs-gumbel':
                gumbel = (-torch.log(1e-8 - torch.log(torch.rand(x.size(0), self.out_features, device=x.device) + 1e-8)))

            g = torch.sigmoid((torch.nn.functional.linear(x, G, bias) + gumbel) / self.tau)
        else:
            g = torch.sigmoid(torch.nn.functional.linear(x, G, bias))

        return g

    def forward(self, x):
        self.stored_input = x

        g_add = self._compute_gate(x, self.G_add, self.bias_add)
        self.stored_gate_add = g_add

        if self.nalu_two_gate:
            g_mul = self._compute_gate(x, self.G_mul, self.bias_mul)
            self.stored_gate_mul = g_mul
            self.writer.add_histogram('gate/add', g_add)
            self.writer.add_histogram('gate/mul', g_mul)
        else:
            g_mul = 1 - g_add
            self.writer.add_histogram('gate', g_add)
            self.writer.add_scalar('gate/mean', torch.mean(g_add), verbose_only=False)

        # a = W x = nac(x)
        a = self.nac_add(x)

        # m = exp(W log(|x| + eps)) = exp(nac(log(|x| + eps)))
        if self.nalu_mul == 'normal':
            m = torch.exp(self.nac_mul(
                torch.log(torch.abs(x) + self.eps)
            ))
        elif self.nalu_mul == 'safe':
            m = torch.exp(self.nac_mul(
                torch.log(torch.abs(x - 1) + 1)
            ))
        elif self.nac_mul == 'max-safe':
            m = torch.exp(self.nac_mul(
                torch.log(torch.relu(x - 1) + 1)
            ))
        elif self.nalu_mul == 'trig':
            m = torch.sinh(self.nac_mul(
                torch.log(x+(x**2+1)**0.5 + self.eps)  # torch.asinh(x) does not exist
            ))
        elif self.nalu_mul == 'mnac':
            m = self.nac_mul(x)
        else:
            raise ValueError(f'Unsupported nalu_mul option ({self.nalu_mul})')

        self.writer.add_histogram('add', a)
        self.writer.add_histogram('mul', m)
        # y = g (*) a + (1 - g) (*) m
        y = g_add * a + g_mul * m

        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}, eps={}, nalu_two_nac={}, nalu_bias={}'.format(
            self.in_features, self.out_features, self.eps, self.nalu_two_nac, self.nalu_bias
        )
