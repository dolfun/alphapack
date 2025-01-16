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
      image_data = np.expand_dims(height_map, axis=0)
      additional_data = np.array(state.normalized_items, dtype=np.float32)
      priors = np.array(evaluation.priors, dtype=np.float32)
      reward = np.array([evaluation.reward], dtype=np.float32)
      samples.append((image_data, additional_data, priors, reward))
  
  np.random.shuffle(samples)
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
  
  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]

@torch.no_grad()
def validate(model, dataloader, device):
  total_loss = 0.0
  total_priors_loss = 0.0
  total_value_loss = 0.0
  for inputs in dataloader:
    inputs = (tensor.to(device) for tensor in list(inputs))
    image_data, additional_data, priors, value = inputs

    predicted_priors, predicted_value = model(image_data, additional_data)
    priors_loss = F.cross_entropy(predicted_priors, priors)
    value_loss = F.mse_loss(predicted_value, value)
    loss = priors_loss + value_loss

    total_loss += loss.item()
    total_priors_loss += priors_loss.item()
    total_value_loss += value_loss.item()

  total_loss /= len(dataloader)
  total_priors_loss /= len(dataloader)
  total_value_loss /= len(dataloader)
  return total_loss, total_priors_loss, total_value_loss

def train_policy_value_network(model, episodes, device):
  samples = prepare_samples(episodes)
  split_ratio = 0.9
  split_count = int(split_ratio * len(samples))
  train_samples, val_samples = samples[:split_count], samples[split_count:]
  train_dataset = ExperienceReplay(train_samples, augment=True)
  val_datset = ExperienceReplay(val_samples)
  train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
  val_dataloader = DataLoader(val_datset, batch_size=2048, shuffle=True)
  print(f'{len(train_dataset)} samples loaded!')

  model.train()
  epochs = 6
  lr = 0.01
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
  for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_priors_loss = 0.0
    epoch_value_loss = 0.0
    for inputs in tqdm(train_dataloader, leave=False):
      inputs = (tensor.to(device) for tensor in list(inputs))
      image_data, additional_data, priors, value = inputs

      predicted_priors, predicted_value = model(image_data, additional_data)
      priors_loss = F.cross_entropy(predicted_priors, priors)
      value_loss = F.mse_loss(predicted_value, value)
      loss = priors_loss + value_loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      epoch_priors_loss += priors_loss.item()
      epoch_value_loss += value_loss.item()

      with open('train.csv', 'a') as f:
        val_loss, _, _ = validate(model, val_dataloader, device)
        f.write(f'{loss.item():.4f},{val_loss:.4f}\n')

    epoch_loss /= len(train_dataloader)
    epoch_priors_loss /= len(train_dataloader)
    epoch_value_loss /= len(train_dataloader)
    print(f'Epoch [{epoch+1}/{epochs}] -> ', end='')
    print(f'Train: {epoch_loss:.4f} = {epoch_priors_loss:.4f} + {epoch_value_loss:.4f}; ', end='')

    val_loss, val_priors_loss, val_value_loss = validate(model, val_dataloader, device)
    print(f'Val: {val_loss:.4f} = {val_priors_loss:.4f} + {val_value_loss:.4f}')