from policy_value_network import PolicyValueNetwork
from bin_packing_solver import State
import bin_packing_solver

import torch.multiprocessing as mp
from tqdm import tqdm
import tempfile
import pickle
import numpy as np
import torch

def load_episodes(file):
  episodes = []
  file.seek(0)
  while True:
    try:
      episode = pickle.load(file)
      episodes.append(episode)
    except EOFError:
      break
  return episodes

def init_worker(_config, model_path, _device):
  global config, model, device

  config = _config
  device = _device

  model = PolicyValueNetwork().to(device)
  model.load_state_dict(torch.load(model_path, weights_only=False))
  model.eval()

@torch.no_grad()
def infer(states):
  global model, device

  image_data = [
    np.array(np.expand_dims(state.height_map, axis=0), dtype=np.float32) / State.bin_height
    for state in states
  ]

  additional_data = [
    np.array(state.normalized_items, dtype=np.float32)
    for state in states
  ]
  
  image_data = torch.tensor(np.stack(image_data, axis=0), device=device)
  additional_data = torch.tensor(np.stack(additional_data, axis=0), device=device)
  priors, value = model.forward(image_data, additional_data)
  priors = torch.softmax(priors, dim=1)
  result = (priors.cpu().numpy(), value.cpu().numpy())
  return result

def generate_episodes_wrapper(episodes_count):
  global config, model, device

  episodes = bin_packing_solver.generate_episodes(
    config.seed,
    config.seed_pool_size,
    episodes_count,
    config.workers_per_process,
    config.move_threshold,
    config.simulations_per_move,
    config.mcts_thread_count,
    config.c_puct,
    config.virtual_loss,
    config.alpha,
    config.batch_size,
    infer
  )

  return episodes

threshold = None
def generate_episodes(config, model_path, device):
  file = tempfile.TemporaryFile()
  initargs = (config, model_path, device)
  with mp.Pool(config.processes, initializer=init_worker, initargs=initargs) as pool:
    steps_count = (config.episodes_per_iteration + config.step_size - 1) // config.step_size
    args = [config.step_size for _ in range(steps_count)]
    it = pool.imap_unordered(generate_episodes_wrapper, args)
    for episodes in tqdm(it, total=steps_count):
      for episode in episodes:
        pickle.dump(episode, file)

  episodes = load_episodes(file)
  rewards = np.array([episode[0].reward for episode in episodes])
  mean_reward = rewards.mean()

  global threshold
  if threshold is None:
    threshold = mean_reward
  if mean_reward > threshold:
    momentum = config.threshold_momentum
    threshold = (1 - momentum) * threshold + momentum * mean_reward

  move_count = 0
  reshaped_rewards = { +1:0, -1:0 }
  for episode in episodes:
    reward = episode[0].reward
    reshaped_reward = +1 if reward > threshold else -1
    reshaped_rewards[reshaped_reward] += 1

    move_count += len(episode)
    for evaluation in episode:
      evaluation.reward = reshaped_reward

  wins = reshaped_rewards[+1]
  losses = reshaped_rewards[-1]
  win_ratio = wins / (wins + losses)
  print(f'{move_count} moves generated!')
  print(f'Average reward: {rewards.mean():.2f} ± {rewards.std():.3f}')
  print(f'Threshold: {threshold:.3f}')
  print(f'Reshaped reward: {wins} wins, {losses} losses ({win_ratio * 100:.1f}%)')

  return episodes