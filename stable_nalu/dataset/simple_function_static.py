
from ._simple_function_abstact import SimpleFunctionDataset

class SimpleFunctionStaticDataset(SimpleFunctionDataset):
    def __init__(self, operation, vector_size=100, **kwargs):
        super().__init__(operation, vector_size, **kwargs)

    def fork(self, input_range=1):
        return super().fork((self._vector_size, ), input_range)
