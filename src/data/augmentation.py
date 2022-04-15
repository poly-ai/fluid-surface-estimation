import numpy as np


# Mirror all frames in a (N, S, H, W) dataset, returning the augmented data
# axis = 2 for vertical (flip row order), 3 for horizontal (flip column order)
def aug_mirror(dataset, axis):
    return np.flip(dataset, axis=axis)


# Rotate 90 degrees CCW, the integer number of times specified
# Rotates 1 time by default
def aug_rotate_90(dataset, times=1):
    return np.rot90(dataset, times, axes=(2, 3))


# Helper function from randomly sampling from a range
def random_sample_range(shape, min, max):

    # Scale the uniform distribution to sample the specified range
    scale = max - min
    shift = min

    return scale * np.random.random_sample(shape) + shift


# Scale each sequence's values by a random constant
def aug_rand_scale(dataset, min_scale_factor=0.5, max_scale_factor=1.5):

    # Sample scale factors and scale dataset
    scale_factors = random_sample_range(
        dataset.shape[0], min_scale_factor, max_scale_factor
    )
    scale_factors = scale_factors.reshape((-1, 1, 1, 1))
    return scale_factors * dataset


# Offset each sequence's values by a random constant
def aug_rand_offset(dataset, min_offset=-1, max_offset=1):

    # Sample offsets and offset dataset
    offsets = random_sample_range(dataset.shape[0], min_offset, max_offset)
    offsets = offsets.reshape((-1, 1, 1, 1))
    return offsets + dataset


# Scale and offsets each sequence by random constants
# clarification: within each sequence, multiply by the same constant
def aug_rand_affine(dataset):
    return aug_rand_offset(aug_rand_scale(dataset))


# Random affine transofmration
# Transforms the input dataset so that everything is between two randomly
# chosen limits, within the range [0,1)
def aug_random_affine_norm(dataset):

    # Generate two limits into which to shift and scale the data
    # lower limits row 0, upper limits row 1
    rand_limits = np.sort(np.random.random_sample(size=(2, dataset.shape[0])), axis=0)

    # Get mins and maxes of each sequence
    seq_mins = dataset.min(axis=(1, 2, 3))
    seq_maxes = dataset.max(axis=(1, 2, 3))

    # Scale
    data_spreads = seq_maxes - seq_mins
    limit_spreads = rand_limits[1, :] - rand_limits[0, :]
    scale_factors = data_spreads / limit_spreads
    dataset /= scale_factors.reshape((-1, 1, 1, 1))

    # Shift
    seq_mins /= scale_factors
    shift_factors = rand_limits[0, :] - seq_mins
    dataset += shift_factors.reshape((-1, 1, 1, 1))

    return dataset


# Randomly chooses pairs of sequences and adds them
# N: number of examples in output dataset
# Returns an augmented dataset the same shape as the input data
def aug_add_random_pairs(dataset, out_size=-1):

    # Handle default value
    if out_size == -1:
        out_size = dataset.shape[0]

    # Get two N-length lists of indicies, from [0, N)
    indices_0 = np.random.randint(0, dataset.shape[0], size=out_size)
    indices_1 = np.random.randint(0, dataset.shape[0], size=out_size)

    return dataset[indices_0] + dataset[indices_1]


def augment(dataset, sum_affine_aug, num_sum_aug, rand_seed):
    orig_dataset_size = dataset.shape[0]
    np.random.seed(rand_seed)

    # Python list containing all augmentations, to contatenate at end
    aug_list = []
    aug_list.append(dataset.copy())

    # Normalized random-affine augmentations
    for _ in range(0, sum_affine_aug):
        aug_list.append(aug_random_affine_norm(dataset))     # Random affine
    dataset = np.vstack(aug_list)

    # Random-pair-sum augmentations, inside normalized random-affine augmentations
    dataset = np.vstack((dataset, aug_random_affine_norm(aug_add_random_pairs(dataset, orig_dataset_size * num_sum_aug))))

    return dataset