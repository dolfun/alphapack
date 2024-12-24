from container_solver import Container, generate_episode
from policy_value_network import PolicyValueNetwork, train_policy_value_network

import os
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device != 'cuda':
  try:
    import torch_directml
    device = torch_directml.device()
  except ImportError:
    pass

import pickle
import argparse
from tqdm import tqdm

def get_test_train_data(file, *, ratio):
  evaluations = []
  file.seek(0)
  while True:
    try:
      evaluations.append(pickle.load(file))
    except EOFError:
      break

  train_count = int(len(evaluations) * ratio)
  return ExperienceReplay(evaluations[:train_count]), ExperienceReplay(evaluations[train_count:])

class ExperienceReplay(Dataset):
  def __init__(self, evaluations):
    self.evaluations = self.__augment_data(evaluations)

  def __augment_data(self, evaluations):
    augmented_data = []
    for image_data, package_data, priors, reward in evaluations:
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
        augmented_data.append((image_data.copy(), package_data, priors, reward))

    return augmented_data

  def __len__(self):
    return len(self.evaluations)

  def __getitem__(self, idx):
    return self.evaluations[idx]
  
def evaluate_with_model(model, containers: list[Container]):
  image_data = []
  additional_data = []
  for container in containers:
    height_map = np.array(container.height_map, dtype=np.float32) / container.height
    image_data.append(np.expand_dims(height_map, axis=0))
    additional_data.append(np.array(container.normalized_packages, dtype=np.float32))
  
  image_data = torch.tensor(np.stack(image_data, axis=0), device=device)
  additional_data = torch.tensor(np.stack(additional_data, axis=0), device=device)
  with torch.no_grad():
    policy, value = model.forward(image_data, additional_data)
    policy = torch.softmax(policy, dim=1)
    result = (policy.cpu().numpy(), value.cpu().numpy())
    return result

def generate_training_data(config, model, episodes_file):
  model.eval()
  evaluate = lambda containers: evaluate_with_model(model, containers)

  rewards = []
  data_points_count = 0
  for _ in tqdm(range(config['games_per_iteration'])):
    episode = generate_episode(
      config['simulations_per_move'], config['thread_count'],
      config['c_puct'], config['virtual_loss'],
      config['batch_size'], evaluate
    )

    data_points_count += len(episode)
    rewards.append(episode[-1].reward)
    
    for evaluation in episode:
      container = evaluation.container
      height_map = np.array(container.height_map, dtype=np.float32) / container.height
      image_data = np.expand_dims(height_map, axis=0)
      additional_data = np.array(container.normalized_packages, dtype=np.float32)
      priors = np.array(evaluation.priors, dtype=np.float32)
      reward = np.array([evaluation.reward], dtype=np.float32)
      pickle.dump((image_data, additional_data, priors, reward), episodes_file)

  rewards = np.array(rewards)
  print(f'{data_points_count} data points generated')
  print(f'Average reward: {rewards.mean():.2} ± {rewards.std():.3f}')

def perform_iteration(config, model):
  # Episodes file
  episodes_file = tempfile.TemporaryFile()

  # Generate Games
  print('GENERATING GAMES:')
  generate_training_data(config, model, episodes_file)
  print()
  
  # Train
  print('TRAINING:')
  train_dataset, test_dataset = get_test_train_data(episodes_file, ratio=0.85)
  trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
  testloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
  train_policy_value_network(model, trainloader, testloader, device)
  print()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iteration_count', type=int, default=1)
  parser.add_argument('--model_path', default='policy_value_network.pth')
  args = parser.parse_args()

  config = {
    'games_per_iteration' : 16,
    'simulations_per_move' : 256,
    'thread_count' : 8,
    'c_puct' : 5.0,
    'virtual_loss' : 3,
    'batch_size' : 4
  }

  # Load model
  model = PolicyValueNetwork().to(device)
  if os.path.exists(args.model_path):
    model.load_state_dict(torch.load(args.model_path, weights_only=False))

  for i in range(args.iteration_count):
    print(f'[{i + 1}/{args.iteration_count}]')
    perform_iteration(config, model)

    # Save model
    torch.save(model.state_dict(), args.model_path)

if __name__ == '__main__':
  main()