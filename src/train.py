import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

def validate_model(model, dataloader, device):
  model.eval()
  with torch.no_grad():
    total_loss = 0.0
    for inputs in dataloader:
      inputs = (tensor.to(device) for tensor in list(inputs))
      image_data, additional_data, priors, reward = inputs

      predicted_priors, predicted_reward = model(image_data, additional_data)
      loss = F.cross_entropy(predicted_priors, priors) + F.mse_loss(predicted_reward, reward)
      total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_policy_value_network(model, train_data, val_data, device):
  train_dataset = ExperienceReplay(train_data)
  val_dataset = ExperienceReplay(val_data)
  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=16)

  model.train()
  learning_rate = 0.0025
  epochs_count = 2
  momentum = 0.9

  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4)
  for epoch in range(epochs_count):
    epoch_loss = 0.0
    for inputs in tqdm(train_loader, leave=False):
      inputs = (tensor.to(device) for tensor in list(inputs))
      image_data, additional_data, priors, reward = inputs

      predicted_priors, predicted_reward = model(image_data, additional_data)
      loss = F.cross_entropy(predicted_priors, priors) + F.mse_loss(predicted_reward, reward)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    val_loss = validate_model(model, val_loader, device)
    print(f'Epoch [{epoch+1}/{epochs_count}], Train: {train_loss:.4f}, Val: {val_loss:.4f}')