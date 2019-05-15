
from ._simple_function_abstact import SimpleFunctionDataset

class SimpleFunctionRecurrentDataset(SimpleFunctionDataset):
    def __init__(self, operation,
                 vector_size=10,
                 min_subset_length=1,
                 max_subset_length=5,
                 min_subset_overlap=1,
                 max_subset_overlap=2, **kwargs):
        super().__init__(operation,
                         vector_size=vector_size,
                         min_subset_length=min_subset_length,
                         max_subset_length=max_subset_length,
                         min_subset_overlap=min_subset_overlap,
                         max_subset_overlap=max_subset_overlap, **kwargs)

    def fork(self, seq_length=10, input_range=1, *args, **kwargs):
        return super().fork((seq_length, self._vector_size), input_range, *args, **kwargs)
