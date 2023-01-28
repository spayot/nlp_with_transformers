from typing import Any

import datasets
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_stratification

Batch = dict[str, list[Any]]


def create_dataset_slices(
    ds: datasets.Dataset,
    slice_sizes: list[int],
    mlb: MultiLabelBinarizer,
    seed: int = None,
) -> list[np.ndarray]:
    """ "returns a list of arrays. the list is of the same length than slice_sizes.
    each array corresponds to a set of training examples to be considered for training.
    the size of the n-th array is defined by the value of the n-th element of slice_sizes"""

    if seed:
        np.random.seed(seed)

    # create list of all indices
    all_indices = np.expand_dims(np.arange(len(ds)), axis=1)

    # initialize
    indices_pool = all_indices  # array with indices available to pool from
    labels = mlb.transform(ds["labels"])
    train_slices, last_k = [], 0

    for i, k in enumerate(slice_sizes):
        (
            indices_pool,
            labels,
            new_slice,
            _,
        ) = iterative_stratification.iterative_train_test_split(
            indices_pool, labels, (k - last_k) / len(labels)
        )

        last_k = k
        if i == 0:
            train_slices.append(new_slice)
        else:
            train_slices.append(np.concatenate([train_slices[-1], new_slice]))

    # add full dataset as last slice:
    train_slices.append(all_indices)
    slice_sizes.append(len(ds))

    return [np.squeeze(train_slice) for train_slice in train_slices]
