
import itertools
import numpy as np
import torch
import torch.utils.data

from ._dataloader import FastDataLoader

class ARITHMETIC_FUNCTIONS:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    def div(a, b):
        return a / b

    def squared(a, b):
        return a * a

    def root(a, b):
        return np.sqrt(a)

class SimpleFunctionDataset:
    def __init__(self, operation, vector_size,
                 seed=None,
                 use_cuda=False,
                 max_size=2**32-1):
        super().__init__()

        self._operation = getattr(ARITHMETIC_FUNCTIONS, operation)
        self._max_size = max_size
        self._vector_size = vector_size
        self._use_cuda = use_cuda
        self._rng = np.random.RandomState(seed)

        self.a_start = self._rng.randint(0, self._vector_size)
        a_size = self._rng.randint(1, self._vector_size - self.a_start + 1)
        self.a_end = self.a_start + a_size

        self.b_start = self._rng.randint(0, self._vector_size)
        b_size = self._rng.randint(1, self._vector_size - self.b_start + 1)
        self.b_end = self.b_start + b_size

    def fork(self, shape, input_range):
        assert shape[-1] == self._vector_size

        rng = np.random.RandomState(self._rng.randint(0, 2**32 - 1))
        return SimpleFunctionDatasetFork(self, shape, input_range, rng)

class SimpleFunctionDatasetFork(torch.utils.data.Dataset):
    def __init__(self, parent, shape, input_range, rng):
        super().__init__()

        self._shape = shape
        self._input_range = input_range
        self._rng = rng

        self._operation = parent._operation
        self._vector_size = parent._vector_size
        self._max_size = parent._max_size
        self._use_cuda = parent._use_cuda

        self._a_start = parent.a_start
        self._a_end = parent.a_end
        self._b_start = parent.b_start
        self._b_end = parent.b_end

    def __getitem__(self, select):
        # Assume select represent a batch_size by using self[0:batch_size]
        batch_size = select.stop - select.start if isinstance(select, slice) else 1

        input_vector = self._rng.uniform(
            low=0,
            high=self._input_range,
            size=(batch_size, ) + self._shape)

        # Compute a and b values
        sum_axies = tuple(range(1, 1 + len(self._shape)))
        a = np.sum(input_vector[..., self._a_start:self._a_end], axis=sum_axies)
        b = np.sum(input_vector[..., self._b_start:self._b_end], axis=sum_axies)

        # Compute result of arithmetic operation
        output_scalar = self._operation(a, b)[:, np.newaxis]

        # If select is an index, just return the content of one row
        if not isinstance(select, slice):
            input_vector = input_vector[0]
            output_scalar = output_scalar[0]

        return (
            torch.tensor(input_vector, dtype=torch.float32),
            torch.tensor(output_scalar, dtype=torch.float32)
        )

    def __len__(self):
        return self._max_size

    def baseline_guess(self, input_vector):
        # Guess and a and b range
        a_start = self._rng.randint(0, self._vector_size)
        a_size = self._rng.randint(1, self._vector_size - a_start + 1)
        a_end = a_start + a_size

        b_start = self._rng.randint(0, self._vector_size)
        b_size = self._rng.randint(1, self._vector_size - b_start + 1)
        b_end = b_start + b_size

        # Compute a and b values
        a = np.sum(input_vector[..., a_start:a_end])
        b = np.sum(input_vector[..., b_start:b_end])

        # Compute result of arithmetic operation
        output_scalar = self._operation(a, b)

        return torch.tensor([output_scalar], dtype=torch.float32)

    def baseline_error(self, batch_size=128):
        squared_error = 0
        for index in range(0, batch_size):
            input_vector, target_scalar = self[index]
            target_guess = self.baseline_guess(input_vector.numpy())
            squared_error += (target_scalar.numpy().item(0) - target_guess.numpy().item(0))**2
        return squared_error / batch_size

    def dataloader(self, batch_size=128):
        return FastDataLoader(self, batch_size, self._use_cuda)
