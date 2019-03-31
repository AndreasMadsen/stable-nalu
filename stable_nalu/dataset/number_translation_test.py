
from nose.tools import *

import torch
import numpy as np

from stable_nalu.dataset import NumberTranslationDataset

def encode_as_string(number):
    return ' '.join(NumberTranslationDataset.encode(number, as_strings=True))

def test_number_encoding():
    assert_equal(encode_as_string(1), 'one')
    assert_equal(encode_as_string(2), 'two')
    assert_equal(encode_as_string(3), 'three')
    assert_equal(encode_as_string(4), 'four')
    assert_equal(encode_as_string(5), 'five')
    assert_equal(encode_as_string(6), 'six')
    assert_equal(encode_as_string(7), 'seven')
    assert_equal(encode_as_string(8), 'eight')
    assert_equal(encode_as_string(9), 'nine')
    assert_equal(encode_as_string(10), 'ten')
    assert_equal(encode_as_string(11), 'eleven')
    assert_equal(encode_as_string(12), 'twelve')
    assert_equal(encode_as_string(13), 'thirteen')
    assert_equal(encode_as_string(14), 'fourteen')
    assert_equal(encode_as_string(15), 'fifteen')
    assert_equal(encode_as_string(16), 'sixteen')
    assert_equal(encode_as_string(17), 'seventeen')
    assert_equal(encode_as_string(18), 'eighteen')
    assert_equal(encode_as_string(19), 'nineteen')
    assert_equal(encode_as_string(20), 'twenty')
    assert_equal(encode_as_string(30), 'thirty')
    assert_equal(encode_as_string(40), 'forty')
    assert_equal(encode_as_string(50), 'fifty')
    assert_equal(encode_as_string(60), 'sixty')
    assert_equal(encode_as_string(70), 'seventy')
    assert_equal(encode_as_string(80), 'eighty')
    assert_equal(encode_as_string(90), 'ninety')
    assert_equal(encode_as_string(100), 'one hundred')
    assert_equal(encode_as_string(230), 'two hundred and thirty')
    assert_equal(encode_as_string(235), 'two hundred and thirty five')
    assert_equal(encode_as_string(35), 'thirty five')
    assert_equal(encode_as_string(119), 'one hundred and nineteen')

def test_train_contains_all_tokens():
    for seed in range(100):
        dataset = NumberTranslationDataset(seed=seed).fork(subset='train')
        tokens_seen = set()
        for x, t in dataset:
            tokens_seen |= set(x.tolist())
        assert_equal(tokens_seen, set(range(29)))

def test_correct_length():
    dataset = NumberTranslationDataset(seed=0)
    dataset_train = dataset.fork(subset='train')
    dataset_valid = dataset.fork(subset='valid')
    dataset_test = dataset.fork(subset='test')
    assert_equal(len(dataset_train), 169)
    assert_equal(len(dataset_valid), 200)
    assert_equal(len(dataset_test), 630)

def test_all_subsets_contains_all_numbers():
    dataset = NumberTranslationDataset(seed=0)
    numbers = set()
    for subset_name in ['train', 'valid', 'test']:
        subset = dataset.fork(subset=subset_name)
        for i in range(len(subset)):
            x, t = subset[i]
            numbers.add(int(t.numpy().item(0)))

    assert_equal(numbers, set(range(1, 1000)))

def test_subsets_are_distinct():
    dataset = NumberTranslationDataset(seed=0)
    dataset_train = dataset.fork(subset='train')
    numbers_train = { dataset_train[i][1].numpy().item(0) for i in range(len(dataset_train)) }
    dataset_valid = dataset.fork(subset='valid')
    numbers_valid = { dataset_valid[i][1].numpy().item(0) for i in range(len(dataset_valid)) }
    dataset_test = dataset.fork(subset='test')
    numbers_test = { dataset_test[i][1].numpy().item(0) for i in range(len(dataset_test)) }

    assert_equal(numbers_train & numbers_valid, set())
    assert_equal(numbers_valid & numbers_test, set())
    assert_equal(numbers_train & numbers_test, set())
