from bin_packing_solver import State
import numpy as np

def additional_data_swap_xy(data):
  swapped = data.copy()
  for i in range(0, len(swapped), State.values_per_item):
    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
  return swapped

def shift_actions(additional_data, image):
  shift_amount = int(np.round(additional_data[0] * State.bin_length)) - 1 # Sketchy
  return np.roll(image, -shift_amount, axis=0)

def rotate_sample(sample):
  image_data, additional_data, priors, reward = sample

  rotated_image_data = np.rot90(image_data[0], k=1).copy()
  rotated_image_data = np.expand_dims(rotated_image_data, axis=0)

  rotated_additional_data = additional_data_swap_xy(additional_data)

  priors_shape = priors.shape
  rotated_priors = np.rot90(priors.reshape(image_data[0].shape), k=1).copy()
  rotated_priors = shift_actions(rotated_additional_data, rotated_priors)
  rotated_priors = rotated_priors.reshape(priors_shape)

  return (rotated_image_data, rotated_additional_data, rotated_priors, reward.copy())

def flip_sample(sample):
  image_data, additional_data, priors, reward = sample

  flipped_image_data = np.flip(image_data[0], axis=0).copy()
  flipped_image_data = np.expand_dims(flipped_image_data, axis=0)

  flipped_additional_data = additional_data.copy()

  priors_shape = priors.shape
  flipped_priors = np.flip(priors.reshape(image_data[0].shape), axis=0).copy()
  flipped_priors = shift_actions(flipped_additional_data, flipped_priors)
  flipped_priors = flipped_priors.reshape(priors_shape)

  return (flipped_image_data, flipped_additional_data, flipped_priors, reward.copy())

def augment_sample(sample):
  samples = []
  samples.append(sample)
  samples.append(rotate_sample(samples[-1]))
  samples.append(rotate_sample(samples[-1]))
  samples.append(rotate_sample(samples[-1]))
  samples.append(flip_sample(sample))
  samples.append(rotate_sample(samples[-1]))
  samples.append(rotate_sample(samples[-1]))
  samples.append(rotate_sample(samples[-1]))
  return samples