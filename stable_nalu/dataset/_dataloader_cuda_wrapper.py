
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
