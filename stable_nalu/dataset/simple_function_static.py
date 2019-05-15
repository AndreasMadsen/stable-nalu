
import numpy as np
from ._simple_function_abstact import SimpleFunctionDataset

class SimpleFunctionStaticDataset(SimpleFunctionDataset):
    def __init__(self, operation,
                 input_size=100,
                 **kwargs):
        super().__init__(operation, input_size,
                         **kwargs)

    def fork(self, sample_range=[1, 2], *args, **kwargs):
        return super().fork((self._input_size, ), sample_range, *args, **kwargs)
