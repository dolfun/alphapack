from policy_value_network import PolicyValueNetwork
from generate import generate_training_data
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
  # Generate Data
  print('GENERATING DATA:')
  data = generate_training_data(config, model_path, device)
  print()

  # Train
  print('TRAINING:')
  model = PolicyValueNetwork().to(device)
  model.load_state_dict(torch.load(model_path, weights_only=False))
  train_policy_value_network(model, data, device)
  torch.save(model.state_dict(), model_path)
  print()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iteration_count', type=int, default=1)
  parser.add_argument('--model_path', default='policy_value_network.pth')
  args = parser.parse_args()

  config = {
    'processes' : 7,
    'games_per_iteration' : 512,
    'simulations_per_move' : 512,
    'thread_count' : 8,
    'c_puct' : 5,
    'virtual_loss' : 3,
    'batch_size' : 8,

    'percentile' : 70,
    'threshold_momentum' : 0.75
  }

  # Create model if does not exist
  if not os.path.exists(args.model_path):
    print('Creating new model!')
    model = PolicyValueNetwork()
    torch.save(model.state_dict(), args.model_path)

  for i in range(args.iteration_count):
    print(f'[{i + 1}/{args.iteration_count}]')
    perform_iteration(config, args.model_path)

if __name__ == '__main__':
  main()