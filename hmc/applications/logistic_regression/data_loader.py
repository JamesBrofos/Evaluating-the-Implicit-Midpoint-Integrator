import os

import numpy as np
from scipy import io


def load_dataset(dataset: str='banana', fold: int=0):
    """Load a logistic regression dataset and a particular train-test fold.

    Args:
        dataset: The name of the dataset to load.
        fold: The fold indicating which entries are to be used for training and
            testing.

    Returns:
        trainset: The training dataset.
        testset: The test dataset.

    """
    data = io.loadmat(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'data', '13-benchmarks.mat'))[dataset][0][0]
    train_index, test_index = data[2][fold] - 1, data[3][fold] - 1
    x_train, x_test, y_train, y_test = (
        data[0][train_index],
        data[0][test_index],
        data[1][train_index].ravel(),
        data[1][test_index].ravel())
    m, s = x_train.mean(0), x_train.std(0)
    x_train = (x_train - m) / s
    x_test = (x_test - m) / s
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
    y_train[y_train < 0.] = 0.
    y_test[y_test < 0.] = 0.
    trainset = (np.asarray(x_train), np.asarray(y_train))
    testset = (np.asarray(x_test), np.asarray(y_test))
    return trainset, testset
