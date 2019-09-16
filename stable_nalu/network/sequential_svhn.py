
import torch
import torchvision
from ..abstract import ExtendedTorchModule
from ..layer import GeneralizedLayer, GeneralizedCell

class SequentialSvhnNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedCell.UNIT_NAMES

    def __init__(self, unit_name, output_size, writer=None,
                 svhn_outputs=1, resnet='resnet18',
                 model_simplification='none',
                 nac_mul='none', eps=1e-7,
                 **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name
        self.output_size = output_size
        self.nac_mul = nac_mul
        self.eps = eps
        self.model_simplification = model_simplification

        # TODO: maybe don't make them learnable, properly zero will surfise here
        if unit_name == 'LSTM':
            self.register_buffer('zero_state_h', torch.Tensor(self.output_size))
            self.register_buffer('zero_state_c', torch.Tensor(self.output_size))
        else:
            self.register_buffer('zero_state', torch.Tensor(self.output_size))

        self.image2label = getattr(torchvision.models, resnet)(
            num_classes=svhn_outputs
        )

        if nac_mul == 'mnac':
            unit_name = unit_name[0:-3] + 'MNAC'
        if self.model_simplification == 'none':
            self.recurent_cell = GeneralizedCell(svhn_outputs, self.output_size,
                                                unit_name,
                                                writer=self.writer,
                                                **kwags)
        self.reset_parameters()

    def _best_init_state(self):
        if self.nac_mul == 'normal' or self.nac_mul == 'mnac':
            return 1
        elif self.nac_mul == 'none':
            return 0

    def reset_parameters(self):
        if self.unit_name == 'LSTM':
            torch.nn.init.constant_(self.zero_state_h, self._best_init_state())
            torch.nn.init.constant_(self.zero_state_c, self._best_init_state())
        else:
            torch.nn.init.constant_(self.zero_state, self._best_init_state())

        # self.image2label.reset_parameters()

        if self.model_simplification == 'none':
            self.recurent_cell.reset_parameters()

    def _forward_trainable_accumulator(self, x):
        y_all = []
        l_all = []

        # Perform recurrent iterations over the input
        if self.unit_name == 'LSTM':
            h_tm1 = (
                self.zero_state_h.repeat(x.size(0), 1),
                self.zero_state_c.repeat(x.size(0), 1)
            )
        else:
            h_tm1 = self.zero_state.repeat(x.size(0), 1)

        for t in range(x.size(1)):
            x_t = x[:, t]
            l_t = self.image2label(x_t)

            if self.nac_mul == 'none' or self.nac_mul == 'mnac':
                h_t = self.recurent_cell(l_t, h_tm1)
            elif self.nac_mul == 'normal':
                h_t = torch.exp(self.recurent_cell(
                    torch.log(torch.abs(l_t) + self.eps),
                    torch.log(torch.abs(h_tm1) + self.eps)
                ))

            y_all.append(h_t[0] if self.unit_name == 'LSTM' else h_t)
            l_all.append(l_t)

            h_tm1 = h_t

        return (
            torch.stack(l_all).transpose(0, 1),
            torch.stack(y_all).transpose(0, 1)
        )

    def _forward_solved_accumulator(self, x):
        y_all = []
        l_all = []

        h_tm1 = self._best_init_state()
        for t in range(x.size(1)):
            x_t = x[:, t]
            l_t = self.image2label(x_t)

            if self.nac_mul == 'normal' or self.nac_mul == 'mnac':
                h_t = h_tm1 * l_t
            elif self.nac_mul == 'none':
                h_t = h_tm1 + l_t

            y_all.append(h_t)
            l_all.append(l_t)

            h_tm1 = h_t

        return (
            torch.stack(l_all).transpose(0, 1),
            torch.stack(y_all).transpose(0, 1)
        )

    def _forward_pass_through(self, x):
        y_all = []
        l_all = []

        for t in range(x.size(1)):
            x_t = x[:, t]
            l_t = self.image2label(x_t)

            y_all.append(l_t)
            l_all.append(l_t)

        return (
            torch.stack(l_all).transpose(0, 1),
            torch.stack(y_all).transpose(0, 1)
        )

    def forward(self, x):
        """Performs recurrent iterations over the input.

        Arguments:
            input: Expected to have the shape [obs, time, channels=1, width, height]
        """
        if self.model_simplification == 'none':
            return self._forward_trainable_accumulator(x)
        elif self.model_simplification == 'solved-accumulator':
            return self._forward_solved_accumulator(x)
        elif self.model_simplification == 'pass-through':
            return self._forward_pass_through(x)
        else:
            raise ValueError('incorrect model_simplification value')

    def extra_repr(self):
        return 'unit_name={}, output_size={}'.format(
            self.unit_name, self.output_size
        )
