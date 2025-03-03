from bin_packing_solver import generate_cut_init_states
from policy_value_network import PolicyValueNetwork
from train import train_policy_value_network
from generate import generate_episodes

from dataclasses import dataclass
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
  pool_size: int
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

def perform_iteration(
    idx: int,
    config: Config,
    model_path: str,
    generate_only: bool):

  # Simulate Games
  print('SIMULATING GAMES:')
  init_states = generate_cut_init_states(config.seed, config.pool_size, 2, 5, 0.0, 1.0, 4096)
  episodes = generate_episodes(init_states, config, model_path, device)
  with open(f'checkpoints/episodes{idx}.bin', 'wb') as file:
    pickle.dump(episodes, file)

  print()

  if generate_only: return

  # Train
  print('TRAINING:')
  model = PolicyValueNetwork().to(device)
  model.load_state_dict(torch.load(model_path, weights_only=False))
  torch.save(model.state_dict(), f'checkpoints/model{idx}.pth')
  train_policy_value_network(model, episodes, device)
  torch.save(model.state_dict(), model_path)
  print()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iteration_count', type=int, default=1)
  parser.add_argument('--generate_only', action='store_true')
  args = parser.parse_args()

  config = Config(
    seed=23894734,
    pool_size=256,
    episodes_per_iteration=1152,
    processes=6,
    step_size=96,
    workers_per_process=32,
    move_threshold=0,
    simulations_per_move=512,
    mcts_thread_count=8,
    batch_size=128,
    c_puct=1.25,
    virtual_loss=1,
    alpha=0.1
  )

  # Create model if does not exist
  model_path = 'policy_value_network.pth'
  if not os.path.exists(model_path):
    print('Creating new model!')
    model = PolicyValueNetwork().to(device)
    torch.save(model.state_dict(), model_path)

  os.makedirs('checkpoints', exist_ok=True)
  for i in range(args.iteration_count):
    print(f'[{i + 1}/{args.iteration_count}]')
    perform_iteration(i + 1, config, model_path, args.generate_only)

if __name__ == '__main__':
  main()