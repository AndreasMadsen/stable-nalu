
from ._simple_function_abstact import SimpleFunctionDataset

class SimpleFunctionStaticDataset(SimpleFunctionDataset):
    def __init__(self, operation, input_size=100, **kwargs):
        super().__init__(operation, (input_size, ), **kwargs)
