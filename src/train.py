from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ExperienceReplay(Dataset):
  def __init__(self, evaluations):
    self.evaluations = self.__augment_data(evaluations)

  def __augment_data(self, evaluations):
    augmented_data = []
    for image_data, additional_data, priors, reward in evaluations:
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

  def __len__(self):
    return len(self.evaluations)

  def __getitem__(self, idx):
    return self.evaluations[idx]

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
  print(f'{8 * len(train_data)} data points loaded!')
  train_loader = DataLoader(ExperienceReplay(train_data), batch_size=128, shuffle=True)
  val_loader = DataLoader(ExperienceReplay(val_data), batch_size=128)

  model.train()
  epochs_count = 4
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
  for epoch in range(epochs_count):
    epoch_loss = 0.0
    epoch_priors_loss = 0.0
    epoch_value_loss = 0.0
    for inputs in tqdm(train_loader, leave=False):
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

    epoch_loss /= len(train_loader)
    epoch_priors_loss /= len(train_loader)
    epoch_value_loss /= len(train_loader)
    val_loss, val_priors_loss, val_value_loss = validate(model, val_loader, device)
    print(f'Epoch [{epoch+1}/{epochs_count}] -> ', end='')
    print(f'Train: {epoch_loss:.4f} = {epoch_priors_loss:.4f} + {epoch_value_loss:.4f}; ', end='')
    print(f'Val: {val_loss:.4f} = {val_priors_loss:.4f} + {val_value_loss:.4f}')