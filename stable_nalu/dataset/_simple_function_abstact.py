
import itertools
import numpy as np
import torch
import torch.utils.data

from ._dataloader_cuda_wrapper import DataLoaderCudaWrapper

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
                 num_workers=0,
                 use_cuda=False,
                 max_size=2**32-1):
        super().__init__()

        self._operation = getattr(ARITHMETIC_FUNCTIONS, operation)
        self._max_size = max_size
        self._vector_size = vector_size
        self._num_workers = num_workers
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

        rngs = [
            np.random.RandomState(self._rng.randint(0, 2**32 - 1))
            for i in range(max(1, self._num_workers))
        ]
        return SimpleFunctionDatasetFork(self, shape, input_range, rngs)

class SimpleFunctionDatasetFork(torch.utils.data.Dataset):
    def __init__(self, parent, shape, input_range, rngs):
        super().__init__()

        self._shape = shape
        self._input_range = input_range
        self._rngs = rngs

        self._operation = parent._operation
        self._max_size = parent._max_size
        self._num_workers = parent._num_workers
        self._use_cuda = parent._use_cuda

        self._a_start = parent.a_start
        self._a_end = parent.a_end
        self._b_start = parent.b_start
        self._b_end = parent.b_end

        self._worker_id = 0

    def _worker_init_fn(self, worker_id):
        self._worker_id = worker_id

    def __getitem__(self, index):
        input_vector = self._rngs[self._worker_id].uniform(
            low=0,
            high=self._input_range,
            size=self._shape)

        # Compute a and b values
        a = np.sum(input_vector[..., self._a_start:self._a_end])
        b = np.sum(input_vector[..., self._b_start:self._b_end])
        # Compute result of arithmetic operation
        output_scalar = self._operation(a, b)

        return (
            torch.tensor(input_vector, dtype=torch.float32),
            torch.tensor([output_scalar], dtype=torch.float32)
        )

    def __len__(self):
        return self._max_size

    def dataloader(self, batch_size=128):
        batcher = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            sampler=torch.utils.data.SequentialSampler(self),
            num_workers=self._num_workers,
            worker_init_fn=self._worker_init_fn)

        if self._use_cuda:
            return DataLoaderCudaWrapper(batcher)
        else:
            return batcher
