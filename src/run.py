from policy_value_network import PolicyValueNetwork
from generate import generate_training_data, load_evaluations
from train import train_policy_value_network

import argparse
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device != 'cuda':
  try:
    import torch_directml
    device = torch_directml.device()
  except ImportError:
    pass

def perform_iteration(config, model_path):
  # Generate Games
  print('GENERATING GAMES:')
  episodes_file = generate_training_data(config, model_path, device)
  print()

  # Retrive training data
  data = load_evaluations(episodes_file)
  split_ratio = 0.9
  split_count = int(split_ratio * len(data))
  train_data, val_data = data[:split_count], data[split_count:]
  
  # Train
  print('TRAINING:')
  print(f'{len(train_data) * 8} data points loaded!')
  model = PolicyValueNetwork().to(device)
  model.load_state_dict(torch.load(model_path, weights_only=False))
  train_policy_value_network(model, train_data, val_data, device)
  torch.save(model.state_dict(), model_path)
  print()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iteration_count', type=int, default=1)
  parser.add_argument('--model_path', default='policy_value_network.pth')
  args = parser.parse_args()

  config = {
    'processes' : 2,
    'games_per_iteration' : 8,
    'simulations_per_move' : 256,
    'thread_count' : 8,
    'c_puct' : 4.5,
    'virtual_loss' : 5,
    'batch_size' : 4
  }

  # Load model
  if not os.path.exists(args.model_path):
    model = PolicyValueNetwork()
    torch.save(model.state_dict(), args.model_path)

  for i in range(args.iteration_count):
    print(f'[{i + 1}/{args.iteration_count}]')
    perform_iteration(config, args.model_path)

if __name__ == '__main__':
  main()