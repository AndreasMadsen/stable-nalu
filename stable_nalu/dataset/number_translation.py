
import os.path as path
import numpy as np
import torch
import torch.utils.data
import torchvision

from ._dataloader import DataLoaderCudaWrapper

id2token = [
    '<pad>',
    'and',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
    'eleven',
    'twelve',
    'thirteen',
    'fourteen',
    'fifteen',
    'sixteen',
    'seventeen',
    'eighteen',
    'nineteen',
    'twenty',
    'thirty',
    'forty',
    'fifty',
    'sixty',
    'seventy',
    'eighty',
    'ninety',
    'hundred'
]
first_ten_tokens = id2token[2:11]
first_twenty_tokens = id2token[2:21]
tens_tokens = id2token[21:29]
pad_token = id2token[0]
and_token = id2token[1]
hundred_token = id2token[-1]

token2id = {
    token: id
    for id, token in enumerate(id2token)
}

class NumberTranslationDataset:
    def __init__(self,
                 num_workers=1,
                 seed=None,
                 use_cuda=False):
        super().__init__()

        self._num_workers = num_workers
        self._use_cuda = use_cuda
        self._rng = np.random.RandomState(seed)

        self._train_samples = []
        self._valid_samples = []
        self._test_samples = []

        # add the first 19 tokens
        missing_train_ids = set(token2id.values()) - set([pad_token])
        for number in range(1, 20):
            ids = self.encode(number)
            self._train_samples.append((ids, number))
            missing_train_ids -= set(ids)

        # add the remaining 999 - 19 tokens
        remaning_numbers = self._rng.permutation(1000 - 20) + 20
        for number in remaning_numbers.tolist():
            ids = self.encode(number)

            # If the sample contains tokens that have not yet been included in
            # the training set, then include it in the training set before
            # any other dataset.
            if len(missing_train_ids) and len(missing_train_ids & set(ids)) > 0:
                    self._train_samples.append((ids, number))
                    missing_train_ids -= set(ids)
            # Because the above filter creates a slight bias, for less unique
            # tokens, fill the test dataset first as this is the biggest. Thus
            # the bias is going to matter the least.
            elif len(self._test_samples) < 630:
                self._test_samples.append((ids, number))
            # In the unlikely case that no occurrences of some token haven't been
            # seen, after 631 observations have been added to the test dataset,
            # continue adding tokens to the validation dataset.
            elif len(self._valid_samples) < 200:
                self._valid_samples.append((ids, number))
            # Adding to the training dataset last, completly ensures that we train
            # over all tokens. Note that it is highly improbable that the order matters,
            # but just in case the seed is bad the training dataset is appended last.
            elif len(self._train_samples) < 169:
                self._train_samples.append((ids, number))
                missing_train_ids -= set(ids)

    @staticmethod
    def encode(number, as_strings=False):
        if number <= 0 or number >= 1000:
            raise ValueError(f'{number} must be between [1, 999]')

        hundreds = number // 100
        tens = (number % 100) // 10
        ones = number % 10

        tokens = []

        if hundreds > 0:
            tokens.append(first_ten_tokens[hundreds - 1])
            tokens.append(hundred_token)

        if len(tokens) > 0 and (tens > 0 or ones > 0):
            tokens.append(and_token)

        if 0 < tens * 10 + ones < 20:
            # from [1, 19]
            tokens.append(first_twenty_tokens[tens * 10 + ones - 1])
        else:
            # from [20, 99]
            if tens > 0:
                tokens.append(tens_tokens[tens - 2])
            if ones > 0:
                tokens.append(first_ten_tokens[ones - 1])

        if as_strings:
            return tokens
        else:
            # pad tokens to have length 6
            tokens += [pad_token] * (5 - len(tokens))
            return np.asarray([token2id[token] for token in tokens], dtype=np.int64)

    def fork(self, subset='train'):
        if subset not in {'train', 'valid', 'test'}:
            raise ValueError(f'subset must be either train, valid or test, it is {subset}')

        if subset == 'train':
            dataset = self._train_samples
        elif subset == 'valid':
            dataset = self._valid_samples
        elif subset == 'test':
            dataset = self._test_samples

        return NumberTranslationDatasetFork(self, dataset)

class NumberTranslationDatasetFork(torch.utils.data.Dataset):
    def __init__(self, parent, dataset):
        super().__init__()

        self._num_workers = parent._num_workers
        self._use_cuda = parent._use_cuda

        self._dataset = dataset

    def __getitem__(self, index):
        x, t = self._dataset[index]

        return (
            torch.tensor(x, dtype=torch.int64),
            torch.tensor([t], dtype=torch.float32)
        )

    def __len__(self):
        return len(self._dataset)

    def dataloader(self, batch_size=64, shuffle=True):
        batcher = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers)

        if self._use_cuda:
            return DataLoaderCudaWrapper(batcher)
        else:
            return batcher
