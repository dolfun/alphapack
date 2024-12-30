from tqdm import tqdm
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ExperienceReplay(Dataset):
  def __init__(self, data, train=False):
    self.data = data
    if train: self.data = self.__oversample_data(self.data)
    self.data = self.__augment_data(self.data)

  def __augment_data(self, data):
    augmented_data = []
    for image_data, additional_data, priors, reward in data:
      image_data = image_data[0]
      reflected = np.flip(image_data, axis=0)
      symmetries = (
        image_data,
        np.rot90(image_data, k=1),
        np.rot90(image_data, k=2),
        np.rot90(image_data, k=-1),
        reflected,
        np.rot90(reflected, k=1),
        np.rot90(reflected, k=2),
        np.rot90(reflected, k=-1)
      )

      for image_data in symmetries:
        image_data = np.expand_dims(image_data, axis=0)
        augmented_data.append((image_data.copy(), additional_data, priors, reward))

    return augmented_data
  
  def __oversample_data(self, data):
    wins = []
    losses = []
    for evaluation in data:
      if evaluation[-1][0] == +1:
        wins.append(evaluation)
      else:
        losses.append(evaluation)
    
    if len(wins) >= len(losses):
      larger, smaller = wins, losses
    else:
      larger, smaller = losses, wins

    if len(smaller) > 0:
      data.extend(random.choices(smaller, k=len(larger) - len(smaller)))

    return data
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

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

def train_policy_value_network(model, data, device):
  split_ratio = 0.9
  split_count = int(split_ratio * len(data))
  train_data, val_data = data[:split_count], data[split_count:]
  train_dataset = ExperienceReplay(train_data, train=True)
  val_datset = ExperienceReplay(val_data)
  train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
  val_dataloader = DataLoader(val_datset, batch_size=2048)
  print(f'{len(train_dataset)} data points loaded!')

  model.train()
  epochs_count = 1
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
  for epoch in range(epochs_count):
    epoch_loss = 0.0
    epoch_priors_loss = 0.0
    epoch_value_loss = 0.0
    counter = 0
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

      with open('train_log.txt', 'a') as f:
        counter += 1
        val_loss, _, _ = validate(model, val_dataloader, device)
        if counter == 1: f.write(f'Epoch [{epoch+1}/{epochs_count}]\n')
        f.write(f'[{counter:<2}/{len(train_dataloader)}] -> Train: {loss.item():.4f}, Val: {val_loss:.4f}\n')
        if counter == len(train_dataloader) and epoch == epochs_count - 1: f.write('\n')

    epoch_loss /= len(train_dataloader)
    epoch_priors_loss /= len(train_dataloader)
    epoch_value_loss /= len(train_dataloader)
    print(f'Epoch [{epoch+1}/{epochs_count}] -> ', end='')
    print(f'Train: {epoch_loss:.4f} = {epoch_priors_loss:.4f} + {epoch_value_loss:.4f}; ', end='')

    val_loss, val_priors_loss, val_value_loss = validate(model, val_dataloader, device)
    print(f'Val: {val_loss:.4f} = {val_priors_loss:.4f} + {val_value_loss:.4f}')