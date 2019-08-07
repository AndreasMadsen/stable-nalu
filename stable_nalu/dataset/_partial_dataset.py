
import torch

class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, full_dataset, offset, length):
        super().__init__()

        self.full_dataset = full_dataset
        self.offset = offset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError()

        return self.full_dataset[self.offset + index]

    def __iter__(self):
        for i in range(self.length):
            yield self[i]
