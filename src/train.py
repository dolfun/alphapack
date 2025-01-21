from bin_packing_solver import State
from augment_sample import augment_sample
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def prepare_samples(episodes):
  samples = []
  for episode in episodes:
    for evaluation in episode:
      state = evaluation.state

      height_map = np.array(state.height_map, dtype=np.float32) / State.bin_height
      feasibility_mask = np.array(state.feasibility_mask, dtype=np.float32)
      image_data = np.stack([height_map, feasibility_mask])
      
      additional_data = np.array(state.normalized_items, dtype=np.float32)

      priors = np.array(evaluation.priors, dtype=np.float32)
      value = np.array([evaluation.reward], dtype=np.float32)

      current_item_shape = (state.items[0].shape.x, state.items[0].shape.y)
      samples.append((image_data, additional_data, priors, value, current_item_shape))

  return samples

class ExperienceReplay(Dataset):
  def __init__(self, samples, augment=False):
    self.samples = samples
    if augment:
      self.samples = [
        augmented_sample
        for sample in self.samples
        for augmented_sample in augment_sample(sample)
      ]

    self.samples = [sample[:-1] for sample in self.samples]

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]

@torch.no_grad()
def validate(model, dataloader, device, loss_scale_factor):
  total_loss = 0.0
  total_priors_loss = 0.0
  total_value_loss = 0.0
  for inputs in dataloader:
    inputs = (tensor.to(device) for tensor in list(inputs))
    image_data, additional_data, priors, value = inputs

    predicted_priors, predicted_value = model(image_data, additional_data)
    priors_loss = F.cross_entropy(predicted_priors, priors)
    value_loss = loss_scale_factor * F.mse_loss(predicted_value, value)
    loss = priors_loss + value_loss

    total_loss += loss.item()
    total_priors_loss += priors_loss.item()
    total_value_loss += value_loss.item()

  total_loss /= len(dataloader)
  total_priors_loss /= len(dataloader)
  total_value_loss /= len(dataloader)
  return total_loss, total_priors_loss, total_value_loss

step_count = 0
def train_policy_value_network(model, episodes, device):
  samples = prepare_samples(episodes)
  np.random.shuffle(samples)

  split_ratio = 0.95
  split_count = int(split_ratio * len(samples))
  train_samples, val_samples = samples[:split_count], samples[split_count:]
  train_dataset = ExperienceReplay(train_samples, augment=True)
  val_datset = ExperienceReplay(val_samples, augment=True) # Should I do that?
  train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
  val_dataloader = DataLoader(val_datset, batch_size=1024)
  print(f'{len(train_dataset)} samples loaded!')

  epochs = 4
  lr = 0.1
  loss_scale_factor = 1.5

  model.train()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
  for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_priors_loss = 0.0
    epoch_value_loss = 0.0
    for inputs in tqdm(train_dataloader, leave=False):
      inputs = (tensor.to(device) for tensor in list(inputs))
      image_data, additional_data, priors, value = inputs

      predicted_priors, predicted_value = model(image_data, additional_data)
      priors_loss = F.cross_entropy(predicted_priors, priors)
      value_loss = loss_scale_factor * F.mse_loss(predicted_value, value)
      loss = priors_loss + value_loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      epoch_priors_loss += priors_loss.item()
      epoch_value_loss += value_loss.item()

      global step_count
      if step_count % 20 == 0:
        val_loss, val_priors_loss, val_value_loss = validate(model, val_dataloader, device, loss_scale_factor)
        with open('train.csv', 'a') as f:
          f.write(f'{step_count}')
          f.write(f',{loss.item():.4f},{priors_loss.item():.4f},{value_loss.item():.4f}')
          f.write(f',{val_loss:.4f},{val_priors_loss:.4f},{val_value_loss:.4f}\n')

      step_count += 1

    epoch_loss /= len(train_dataloader)
    epoch_priors_loss /= len(train_dataloader)
    epoch_value_loss /= len(train_dataloader)
    print(f'Epoch [{epoch+1}/{epochs}] -> ', end='')
    print(f'Train: {epoch_loss:.4f} = {epoch_priors_loss:.4f} + {epoch_value_loss:.4f}; ', end='')

    val_loss, val_priors_loss, val_value_loss = validate(model, val_dataloader, device, loss_scale_factor)
    print(f'Val: {val_loss:.4f} = {val_priors_loss:.4f} + {val_value_loss:.4f}')