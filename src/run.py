from policy_value_network import PolicyValueNetwork
from train import train_policy_value_network
from generate import generate_episodes

from dataclasses import dataclass
from copy import copy
import argparse
import pickle
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
  move_threshold: int
  simulations_per_move: int
  mcts_thread_count: int
  batch_size: int
  c_puct: float
  virtual_loss: int
  alpha: float

def perform_iteration(config: Config, model_path: str, generate_only: bool):
  # Simulate Games
  print('SIMULATING GAMES:')
  episodes = generate_episodes(config, model_path, device)
  with open('episodes_train.bin', 'wb') as f:
    pickle.dump(episodes, f)

  print()

  if generate_only: return

  # Train
  print('TRAINING:')
  model = PolicyValueNetwork().to(device)
  model.load_state_dict(torch.load(model_path, weights_only=False))
  train_policy_value_network(model, episodes, device)
  torch.save(model.state_dict(), model_path)
  print()

  # Evaluation
  print('EVALUATING:')
  eval_config = copy(config)
  eval_config.episodes_per_iteration = 256
  eval_config.move_threshold = 0
  eval_config.alpha = 0

  episodes = generate_episodes(eval_config, model_path, device, leave_progress_bar=False)
  with open('episodes_test.bin', 'wb') as f:
    pickle.dump(episodes, f)

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
    episodes_per_iteration=2048,
    processes=4,
    step_size=32,
    workers_per_process=24,
    move_threshold=4,
    simulations_per_move=800,
    mcts_thread_count=8,
    batch_size=32,
    c_puct=2.5,
    virtual_loss=3,
    alpha=0.1
  )

  # Create model if does not exist
  if not os.path.exists(args.model_path):
    print('Creating new model!')
    model = PolicyValueNetwork().to(device)
    torch.save(model.state_dict(), args.model_path)

  for i in range(args.iteration_count):
    print(f'[{i + 1}/{args.iteration_count}]')
    perform_iteration(config, args.model_path, args.generate_only)

if __name__ == '__main__':
  main()