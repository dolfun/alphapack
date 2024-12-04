from container_solver import Container, generate_episode
from policy_value_network import PolicyValueNetwork, train_policy_value_network
from package_utils import random_package, normalize_packages

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import requests
import pickle
import argparse
from tqdm import tqdm

class ExperienceReplay(Dataset):
  def __init__(self, episodes_file_path):
    self.episodes = []
    with open(episodes_file_path, 'rb') as f:
      while True:
        try:
          self.episodes.append(pickle.load(f))
        except EOFError:
          break

  def __len__(self):
    return len(self.episodes)
  
  def __getitem__(self, idx):
    return self.episodes[idx]

def save_episode(episode, episodes_file):
  container = episode.container
  priors = episode.priors
  reward = episode.reward

  height_map = np.array(container.height_map, dtype=np.float32) / container.height
  height_map = np.expand_dims(height_map, axis=0)
  packages_data = normalize_packages(container.packages)
  priors = np.array(priors, dtype=np.float32)
  reward = np.array([reward], dtype=np.float32)
  pickle.dump((height_map, packages_data, priors, reward), episodes_file)

def generate_training_data(games_per_iteration, simulations_per_move, episodes_file):
  rewards = []
  episodes_count = 0
  for _ in tqdm(range(games_per_iteration)):
    container_height = 24
    packages = [random_package() for _ in range(Container.action_count)]
    container = Container(container_height, packages)

    episodes = generate_episode(container, simulations_per_move)
    episodes_count += len(episodes)
    rewards.append(episodes[-1].reward)
    
    for episode in episodes:
      save_episode(episode, episodes_file)

  print(f'{episodes_count} episodes generated')
  rewards = np.array(rewards)
  print(f'Average reward: {rewards.mean():.2} ± {rewards.std():.2}')

def perform_iteration(model_path, worker_addresses, episodes_file_path, generate_only=False):
  # Create model if it does not exist
  if not os.path.exists(model_path):
    policy_value_network = PolicyValueNetwork()
    torch.save(policy_value_network.state_dict(), model_path)

  # Uploaad model to all workers
  with open(model_path, 'rb') as model:
    for address in worker_addresses:
      model.seek(0)
      files = { 'file': model }
      response = requests.post('http://' + address + '/policy_value_upload', files=files)
      if response.text != 'success':
        raise Exception(f'Model upload failed on worker: {address}')

  # Generate Games
  print('Generating Games:')
  with open(episodes_file_path, 'w'):
    pass

  games_per_iteration = 16
  simulations_per_move = 16
  with open(episodes_file_path, 'ab') as file:
    generate_training_data(games_per_iteration, simulations_per_move, file)
  print()

  if generate_only:
    return
  
  # Train
  print('Training:')
  dataset = ExperienceReplay(episodes_file_path)
  dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
  model = PolicyValueNetwork()
  model.load_state_dict(torch.load(model_path, weights_only=True))
  train_policy_value_network(model, dataloader)
  torch.save(model.state_dict(), model_path)
  print()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iteration_count', type=int, default=1)
  parser.add_argument('--model_path', default='policy_value_network.pth')
  parser.add_argument('--worker_addresses', default='127.0.0.1:8000')
  parser.add_argument('--generate_only', type=bool, default=False)
  args = parser.parse_args()

  worker_addresses = args.worker_addresses.split(';')
  for i in range(args.iteration_count):
    print(f'ITERATION: [{i + 1}/{args.iteration_count}]')
    perform_iteration(args.model_path, worker_addresses, 'episodes.bin', args.generate_only)

if __name__ == '__main__':
  main()
