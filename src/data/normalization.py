import numpy as np


# Normalize dataset such that all sequences have min value 0.0, max value 1.0
def normalize(dataset, lower_lim=0.0, upper_lim=1.0):

    seq_mins = dataset.min(axis=(1, 2, 3))
    seq_maxes = dataset.max(axis=(1, 2, 3))

    dataset -= seq_mins.reshape((-1, 1, 1, 1))
    dataset /= (seq_maxes - seq_mins).reshape((-1, 1, 1, 1))

    return dataset


# Normalize only the sequences in the data that have value outside range [0, 1)
# Normalizes these sequences to have min value 0.0, max value 1.0
def normalize_only_outliers(dataset, lower_lim=0.0, upper_lim=1.0):

    # Scale and offset each sequence so that all values are within [0,1)
    seq_mins = dataset.min(axis=(1, 2, 3))
    seq_maxes = dataset.max(axis=(1, 2, 3))

    # Limit normalization only to waves that are out of the range [0,1)
    active = np.logical_or(
        np.less(seq_mins, lower_lim), np.greater(seq_maxes, upper_lim)
    )

    dataset[active] -= seq_mins[active].reshape((-1, 1, 1, 1))
    dataset[active] /= (seq_maxes - seq_mins)[active].reshape((-1, 1, 1, 1))

    return dataset
