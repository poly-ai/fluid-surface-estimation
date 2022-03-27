import numpy as np

# Mirror all frames in a (N, S, H, W) dataset, returning the augmented data
# axis = 2 for vertical (flip row order), 3 for horizontal (flip column order)
def aug_mirror(dataset, axis):
  return np.flip(dataset, axis=axis)


# Rotate 90 degrees CCW, the integer number of times specified 
# Rotates 1 time by default
def aug_rotate_90(dataset, times=1):
  return np.rot90(dataset, times, axes=(2,3))


# Helper function from randomly sampling from a range
def random_sample_range(shape, min, max):

  # Scale the uniform distribution to sample the specified range
  scale = max - min
  shift = min

  return scale * np.random.random_sample(shape) + shift


# Scale each sequence's values by a random constant
def aug_rand_scale(dataset, min_scale_factor=0.5, max_scale_factor=1.5):

  # Sample scale factors and scale dataset
  scale_factors = random_sample_range(dataset.shape[0], min_scale_factor, max_scale_factor)
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
  return(aug_rand_offset(aug_rand_scale(dataset)))


# Randomly chooses pairs of sequences and adds them
# Returns an augmented dataset the same shape as the input data
def aug_add_random_pairs(dataset):
  # Get two N-length lists of indicies, from [0, N)
  indices_0 = np.random.randint(0, dataset.shape[0], size=(dataset.shape[0]))
  indices_1 = np.random.randint(0, dataset.shape[0], size=(dataset.shape[0]))

  return dataset[indices_0] + dataset[indices_1]