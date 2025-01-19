from bin_packing_solver import State
import numpy as np

def additional_data_swap_xy(data):
  for i in range(0, len(data), State.values_per_item):
    data[i], data[i + 1] = data[i + 1], data[i]
  return data

def shift_loading_positions(image, current_item_shape):
  shift_amount = current_item_shape[0] - 1
  return np.roll(image, -shift_amount, axis=0)

def rotate_sample(sample):
  image_data, additional_data, priors, reward, current_item_shape = sample

  current_item_shape = (current_item_shape[1], current_item_shape[0])

  rotated_height_map = np.rot90(image_data[0].copy(), k=1)
  rotated_feasibility_mask = np.rot90(image_data[1].copy(), k=1)
  shifted_feasibility_mask = shift_loading_positions(rotated_feasibility_mask, current_item_shape)
  rotated_image_data = np.stack([rotated_height_map, shifted_feasibility_mask])

  rotated_additional_data = additional_data_swap_xy(additional_data.copy())

  priors = priors.copy().reshape((State.bin_length, State.bin_length))
  rotated_priors = np.rot90(priors, k=1)
  shifted_priors = shift_loading_positions(rotated_priors, current_item_shape)
  shifted_priors = shifted_priors.flatten()

  reward = reward.copy()

  return (rotated_image_data, rotated_additional_data, shifted_priors, reward, current_item_shape)

def flip_sample(sample):
  image_data, additional_data, priors, reward, current_item_shape = sample

  flipped_height_map = np.flip(image_data[0].copy(), axis=0)
  flipped_feasibility_mask = np.flip(image_data[1].copy(), axis=0)
  shifted_feasibility_mask = shift_loading_positions(flipped_feasibility_mask, current_item_shape)
  flipped_image_data = np.stack([flipped_height_map, shifted_feasibility_mask])

  flipped_additional_data = additional_data.copy()

  priors = priors.copy().reshape((State.bin_length, State.bin_length))
  flipped_priors = np.flip(priors, axis=0)
  shifted_priors = shift_loading_positions(flipped_priors, current_item_shape)
  shifted_priors = shifted_priors.flatten()

  reward = reward.copy()

  return (flipped_image_data, flipped_additional_data, shifted_priors, reward, current_item_shape)

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