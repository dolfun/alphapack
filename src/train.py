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

step_count = 0
def train_policy_value_network(model, episodes, device):
  samples = prepare_samples(episodes)
  dataset = ExperienceReplay(samples, augment=True)
  dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
  print(f'{len(dataset)} samples loaded!')

  epochs = 2
  lr = 0.1
  loss_scale_factor = 1.5

  model.train()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
  train_log_file = open('train_log.csv', 'a')
  for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_priors_loss = 0.0
    epoch_value_loss = 0.0
    for inputs in tqdm(dataloader, leave=False):
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
      step_count += 1
      train_log_file.write(f'{step_count}')
      train_log_file.write(f',{loss.item():.4f},{priors_loss.item():.4f},{value_loss.item():.4f}\n')

    epoch_loss /= len(dataloader)
    epoch_priors_loss /= len(dataloader)
    epoch_value_loss /= len(dataloader)
    print(f'Epoch [{epoch+1:2}/{epochs}] -> ', end='')
    print(f'Loss: {epoch_loss:.4f} = {epoch_priors_loss:.4f} + {epoch_value_loss:.4f}')