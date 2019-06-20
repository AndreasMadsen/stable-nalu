
import os.path as path
import numpy as np
import torch
import torch.utils.data
import torchvision
from typing import Tuple, NamedTuple, Union

from ._dataloader import DataLoaderCudaWrapper

class ItemShape(NamedTuple):
    input: Tuple[Union[None, int], ...]
    target: Tuple[Union[None, int], ...]

class OPERATIONS:
    @staticmethod
    def sum(seq):
        return np.sum(seq).reshape(1)

    @staticmethod
    def cumsum(seq):
        return np.cumsum(seq).reshape(-1, 1)

    @staticmethod
    def prod(seq):
        return np.prod(seq).reshape(1)

    @staticmethod
    def cumprod(seq):
        return np.cumprod(seq).reshape(-1, 1)

    @staticmethod
    def count(seq):
        unique, counts = np.unique(seq, return_counts=True)
        counts_all = np.zeros(10)
        counts_all[unique] = counts
        return counts_all.reshape(10)

THIS_DIR = path.dirname(path.realpath(__file__))
DATA_DIR = path.join(THIS_DIR, 'data')

class SequentialMnistDataset:
    def __init__(self, operation,
                 num_workers=1,
                 mnist_digits=[0,1,2,3,4,5,6,7,8,9],
                 seed=None,
                 use_cuda=False):
        super().__init__()

        self._operation = getattr(OPERATIONS, operation)
        self._num_workers = num_workers
        self._use_cuda = use_cuda
        self._rng = np.random.RandomState(seed)
        self._mnist_digits = set(mnist_digits)

    def get_item_shape(self):
        if self._operation == OPERATIONS.sum:
            return ItemShape((None, 28, 28), (1, ))
        elif self._operation == OPERATIONS.cumsum:
            return ItemShape((None, 28, 28), (None, 1))
        elif self._operation == OPERATIONS.prod:
            return ItemShape((None, 28, 28), (1, ))
        elif self._operation == OPERATIONS.cumprod:
            return ItemShape((None, 28, 28), (None, 1))
        else:
            return ItemShape((None, 28, 28), (10, ))

    def fork(self, seq_length=10, subset='train', seed=None):
        if subset not in {'train', 'test'}:
            raise ValueError(f'subset must be either train or test, it is {subset}')

        rng = np.random.RandomState(self._rng.randint(0, 2**32 - 1) if seed is None else seed)
        return SequentialMnistDatasetFork(self, seq_length, subset, rng)

class SequentialMnistDatasetFork(torch.utils.data.Dataset):
    def __init__(self, parent, seq_length, subset, rng):
        super().__init__()

        self._operation = parent._operation
        self._num_workers = parent._num_workers
        self._use_cuda = parent._use_cuda
        self._mnist_digits = parent._mnist_digits

        self._subset = subset
        self._seq_length = seq_length
        self._rng = rng

        self._dataset = torchvision.datasets.MNIST(
            root=DATA_DIR,
            train=subset == 'train',
            download=True,
            # Transform is from https://github.com/pytorch/examples/blob/master/mnist/main.py
            # which is what is also used in the NALU paper
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        self._index_mapping = self._rng.permutation([
            i for (i, (x, t)) in enumerate(self._dataset) if t in self._mnist_digits
        ])

    def __getitem__(self, index):
        mnist_images = []
        mnist_targets = []
        for mnist_index in range(index * self._seq_length, (index + 1) * self._seq_length):
            image, target = self._dataset[self._index_mapping[mnist_index]]
            mnist_images.append(image)  # image.size() = [1, 28, 28]
            mnist_targets.append(target)

        data = torch.stack(mnist_images)  # data.size() = [seq_length, 1, 28, 28]
        target = self._operation(np.stack(mnist_targets))

        return (
            data,
            torch.tensor(target, dtype=torch.float32)
        )

    def __len__(self):
        return len(self._index_mapping) // self._seq_length

    def dataloader(self, batch_size=64, shuffle=True):
        batcher = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers)

        if self._use_cuda:
            return DataLoaderCudaWrapper(batcher)
        else:
            return batcher
