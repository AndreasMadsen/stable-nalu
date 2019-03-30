
class DataLoaderCudaWrapper:
    def __init__(self, batcher):
        self._batcher = batcher

    def __iter__(self):
        batcher = iter(self._batcher)
        return map(lambda values: (value.cuda() for value in values), batcher)
