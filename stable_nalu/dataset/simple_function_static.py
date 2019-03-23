
import numpy as np
import torch
import torch.utils.data

ARITHMETIC_FUNCTIONS = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a / b,
    'squared': lambda a, b: a * a,
    'root': lambda a, b: torch.sqrt(a)
}

class SimpleFunctionStaticDataset(torch.utils.data.Dataset):
    def __init__(self, operation, input_range=5,
                 input_size=100, max_size=2**32-1, seed=None):
        self._operation = ARITHMETIC_FUNCTIONS[operation]
        self._input_range = input_range
        self._input_size = input_size
        self._rng = np.random.RandomState(seed)
        self._max_size = max_size

        self.a_start = self._rng.randint(0, self._input_size)
        a_size = self._rng.randint(1, self._input_size - self.a_start + 1)
        self.a_end = self.a_start + a_size

        self.b_start = self._rng.randint(0, self._input_size)
        b_size = self._rng.randint(1, self._input_size - self.b_start + 1)
        self.b_end = self.b_start + b_size

        self._last_index = -1

    def __getitem__(self, index):
        if (self._last_index + 1 != index):
            raise RuntimeError('expected incrementing index in SimpleFunction Dataset')
        self._last_index += 1

        input_vector = self._rng.uniform(
            low=-self._input_range,
            high=self._input_range,
            size=self._input_size)

        # COmpute a and b values
        a = np.sum(input_vector[self.a_start:self.a_end])
        b = np.sum(input_vector[self.b_start:self.b_end])
        # Compute result of arithmetic operation
        output_scalar = self._operation(a, b)

        return (
            torch.tensor(input_vector, dtype=torch.float32),
            torch.tensor([output_scalar], dtype=torch.float32)
        )

    def __len__(self):
        return self._max_size
