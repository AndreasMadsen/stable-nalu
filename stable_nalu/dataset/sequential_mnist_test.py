
from nose.tools import *

import torch
import numpy as np

from stable_nalu.dataset import SequentialMnistDataset

def test_seed_gives_consistent_output():
    dataset_a = SequentialMnistDataset(
        operation='count',
        seed=0
    ).fork(seq_length=1, subset='train')

    dataset_b = SequentialMnistDataset(
        operation='count',
        seed=0
    ).fork(seq_length=1, subset='train')

    assert_equal(len(dataset_a), len(dataset_b))

    for observation_i in range(100):
        x_a, t_a = dataset_a[observation_i]
        x_b, t_b = dataset_b[observation_i]
        np.testing.assert_almost_equal(x_a.numpy(), x_b.numpy())
        np.testing.assert_almost_equal(t_a.numpy(), t_b.numpy())
        assert_equal(x_a.size(), (1, 1, 28, 28))
        assert_equal(x_b.size(), (1, 1, 28, 28))

def test_train_is_distinct_from_test():
    dataset_train = SequentialMnistDataset(
        operation='count',
        seed=0
    ).fork(seq_length=1, subset='train')

    dataset_test = SequentialMnistDataset(
        operation='count',
        seed=0
    ).fork(seq_length=1, subset='test')

    assert_not_equal(len(dataset_train), len(dataset_test))

    for observation_i in range(100):
        x_a, t_a = dataset_train[observation_i]
        x_b, t_b = dataset_test[observation_i]
        assert not np.allclose(x_a.numpy(), x_b.numpy())
        assert_equal(x_a.size(), (1, 1, 28, 28))
        assert_equal(x_b.size(), (1, 1, 28, 28))

def test_count_dataset():
    for seq_length in [1, 10, 100]:
        dataset_labels = SequentialMnistDataset(
            operation='sum',
            seed=0
        ).fork(seq_length=1, subset='train')

        dataset_count = SequentialMnistDataset(
            operation='count',
            seed=0
        ).fork(seq_length=seq_length, subset='train')

        assert_equal(len(dataset_count), len(dataset_labels) // seq_length)

        for count_i in range(5):
            x_count, t_count = dataset_count[count_i]
            assert_equal(x_count.size(), (seq_length, 1, 28, 28))
            assert_equal(t_count.size(), (10, ))
            t_count_expected = np.zeros((10, ))

            for time_i in range(seq_length):
                x_label, t_label = dataset_labels[count_i * seq_length + time_i]
                t_count_expected[t_label.numpy().astype('int8')] += 1

                np.testing.assert_almost_equal(
                    x_count[time_i].numpy(),
                    x_label.numpy().reshape(1, 28, 28)
                )

            np.testing.assert_almost_equal(t_count.numpy(), t_count_expected)

def test_sum_dataset():
    for seq_length in [1, 10, 100]:
        dataset_labels = SequentialMnistDataset(
            operation='sum',
            seed=0
        ).fork(seq_length=1, subset='train')

        dataset_count = SequentialMnistDataset(
            operation='sum',
            seed=0
        ).fork(seq_length=seq_length, subset='train')

        assert_equal(len(dataset_count), len(dataset_labels) // seq_length)

        for count_i in range(5):
            x_count, t_sum = dataset_count[count_i]
            assert_equal(x_count.size(), (seq_length, 1, 28, 28))
            assert_equal(t_sum.size(), (1, ))
            t_sum_expected = np.zeros((1, ))

            for time_i in range(seq_length):
                x_label, t_label = dataset_labels[count_i * seq_length + time_i]
                t_sum_expected += t_label.numpy()

                np.testing.assert_almost_equal(
                    x_count[time_i].numpy(),
                    x_label.numpy().reshape(1, 28, 28)
                )

            np.testing.assert_almost_equal(t_sum.numpy(), t_sum_expected)
