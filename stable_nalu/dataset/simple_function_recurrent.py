
from ._simple_function_abstact import SimpleFunctionDataset

class SimpleFunctionRecurrentDataset(SimpleFunctionDataset):
    def __init__(self, operation, input_size=10, time_length=10, **kwargs):
        super().__init__(operation, (time_length, input_size), **kwargs)
