from policy_value_network import PolicyValueNetwork
from generate import generate_training_data
from train import train_policy_value_network

import os
import argparse
import tempfile
import pickle
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device != 'cuda':
  try:
    import torch_directml
    device = torch_directml.device()
  except ImportError:
    pass

def perform_iteration(config, model):
  # Episodes file
  episodes_file = tempfile.TemporaryFile()

  # Generate Games
  print('GENERATING GAMES:')
  generate_training_data(config, model, device, episodes_file)
  print()

  # Retrive training data
  episodes_file.seek(0)
  data = []
  while True:
    try:
      data.append(pickle.load(episodes_file))
    except EOFError:
      break

  split_ratio = 0.9
  split_count = int(split_ratio * len(data))
  train_data, validate_data = data[:split_count], data[split_count:]
  
  # Train
  print('TRAINING:')
  train_policy_value_network(model, train_data, validate_data, device)
  print()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iteration_count', type=int, default=1)
  parser.add_argument('--model_path', default='policy_value_network.pth')
  args = parser.parse_args()

  config = {
    'games_per_iteration' : 1024,
    'simulations_per_move' : 256,
    'thread_count' : 8,
    'c_puct' : 4.5,
    'virtual_loss' : 5,
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