
class DataLoaderCudaWrapper:
    def __init__(self, batcher):
        self._batcher = batcher

    def __getattr__(self, name):
        return getattr(self._batcher, name)

    def __iter__(self):
        batcher = iter(self._batcher)
        return map(lambda values: (value.cuda() for value in values), batcher)

    def __len__(self):
        return len(self._batcher)

class FastDataLoader:
    def __init__(self, dataset, batch_size, use_cuda):
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    def __iter__(self):
        for i in range(len(self)):
            values = self.dataset[i * self.batch_size: min(len(self.dataset), (1 + i)*self.batch_size)]
            if self.use_cuda:
                yield tuple(value.cuda() for value in values)
            else:
                yield values

    def __len__(self):
        return len(self.dataset) // self.batch_size
