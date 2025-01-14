from policy_value_network import PolicyValueNetwork
from train import train_policy_value_network
from generate import generate_episodes

from dataclasses import dataclass
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

@dataclass
class Config:
  seed: int
  seed_pool_size: int
  episodes_per_iteration: int
  processes: int
  step_size: int
  workers_per_process: int
  simulations_per_move: int
  mcts_thread_count: int
  batch_size: int
  c_puct: float
  virtual_loss: int
  threshold_momentum: float

def perform_iteration(config, model_path, generate_only):
  # Generate Episodes
  print('GENERATING EPISODES:')
  episodes = generate_episodes(config, model_path, device)
  print()

  with open('checkpoint/episodes.bin', 'wb') as f:
    import pickle
    pickle.dump(episodes, f)

  if generate_only: return

  # Train
  print('TRAINING:')
  model = PolicyValueNetwork().to(device)
  model.load_state_dict(torch.load(model_path, weights_only=False))
  train_policy_value_network(model, episodes, device)
  torch.save(model.state_dict(), model_path)
  print()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iteration_count', type=int, default=1)
  parser.add_argument('--model_path', default='policy_value_network.pth')
  parser.add_argument('--generate_only', action='store_true')
  args = parser.parse_args()

  config = Config(
    seed=2389473453,
    seed_pool_size=2048,
    episodes_per_iteration=512,
    processes=4,
    step_size=32,
    workers_per_process=16,
    simulations_per_move=256,
    mcts_thread_count=8,
    batch_size=32,
    c_puct=5,
    virtual_loss=3,
    threshold_momentum=0.75
  )

  # Create model if does not exist
  if not os.path.exists(args.model_path):
    print('Creating new model!')
    model = PolicyValueNetwork()
    torch.save(model.state_dict(), args.model_path)

  for i in range(args.iteration_count):
    print(f'[{i + 1}/{args.iteration_count}]')
    perform_iteration(config, args.model_path, args.generate_only)

if __name__ == '__main__':
  main()