
import torch
from ..abstract import ExtendedTorchModule
from ..layer import GeneralizedLayer, GeneralizedCell

class NumberTranslationNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedCell.UNIT_NAMES

    def __init__(self, unit_name,
                 embedding_size=2,  # 1 for the number, 1 for the gate ?
                 hidden_size=2,  # 1 for the number, 1 for the gate ?
                 dictionary_size=30,
                 writer=None,
                 **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dictionary_size = dictionary_size

        self.register_buffer('lstm_zero_state_h', torch.Tensor(hidden_size))
        self.register_buffer('lstm_zero_state_c', torch.Tensor(hidden_size))
        self.register_buffer('output_zero_state', torch.Tensor(1))

        self.embedding = torch.nn.Embedding(dictionary_size, embedding_size)
        self.lstm_cell = torch.nn.LSTMCell(embedding_size, hidden_size)
        self.output_cell = GeneralizedCell(hidden_size, 1,
                                        unit_name,
                                        writer=self.writer,
                                        name='recurrent_output',
                                        **kwags)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.lstm_zero_state_h)
        torch.nn.init.zeros_(self.lstm_zero_state_c)
        torch.nn.init.zeros_(self.output_zero_state)

        self.embedding.reset_parameters()
        self.lstm_cell.reset_parameters()
        self.output_cell.reset_parameters()

    def forward(self, x):
        """Performs recurrent iterations over the input.

        Arguments:
            input: Expected to have the shape [obs, time]
        """
        # Perform recurrent iterations over the input
        h_1_tm1 = self.lstm_zero_state_h.repeat(x.size(0), 1)
        c_1_tm1 = self.lstm_zero_state_c.repeat(x.size(0), 1)
        h_2_tm1 = self.output_zero_state.repeat(x.size(0), 1)

        for t in range(x.size(1)):
            x_t = x[:, t]
            h_0_t = self.embedding(x_t)
            h_1_t, c_1_t = self.lstm_cell(h_0_t, (h_1_tm1, c_1_tm1))
            h_2_t = self.output_cell(h_1_t, h_2_tm1)

            # Just use previuse results if x is a <pad> token
            h_2_t = torch.where(x[:, t].view(-1, 1) == 0, h_2_tm1, h_2_t)

            # Prepear for next iterations
            h_1_tm1 = h_1_t
            c_1_tm1 = c_1_t
            h_2_tm1 = h_2_t

        return h_2_t

    def extra_repr(self):
        return 'unit_name={}, embedding_size={}, hidden_size={}, dictionary_size={}'.format(
            self.unit_name, self.embedding_size, self.hidden_size, self.dictionary_size
        )
