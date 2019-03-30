
from ._simple_function_abstact import SimpleFunctionDataset

class SimpleFunctionRecurrentDataset(SimpleFunctionDataset):
    def __init__(self, operation, vector_size=10, **kwargs):
        super().__init__(operation, vector_size, **kwargs)

    def fork(self, seq_length=10, input_range=1):
        return super().fork((seq_length, self._vector_size), input_range)
