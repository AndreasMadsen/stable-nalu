
import math
from ._simple_function_abstact import SimpleFunctionDataset

class SimpleFunctionStaticDataset(SimpleFunctionDataset):
    def __init__(self, operation,
                 vector_size=100,
                 min_subset_length=0.15,
                 max_subset_length=0.50,
                 min_subset_overlap=0.05,
                 max_subset_overlap=0.15,
                 **kwargs):
        super().__init__(operation,
                         vector_size=vector_size,
                         min_subset_length=math.floor(vector_size * min_subset_length),
                         max_subset_length=math.floor(vector_size * max_subset_length),
                         min_subset_overlap=math.floor(vector_size * min_subset_overlap),
                         max_subset_overlap=math.floor(vector_size * max_subset_overlap),
                         **kwargs)

    def fork(self, input_range=1):
        return super().fork((self._vector_size, ), input_range)
