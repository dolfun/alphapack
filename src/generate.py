from policy_value_network import PolicyValueNetwork
from bin_packing_solver import generate_episodes

import torch.multiprocessing as mp
from tqdm import tqdm
import tempfile
import pickle
import numpy as np
import torch

def dump_episode(episode, file):
  episode_data = []
  for evaluation in episode:
    state = evaluation.state
    height_map = np.array(state.height_map, dtype=np.float32) / state.bin_height
    image_data = np.expand_dims(height_map, axis=0)
    additional_data = np.array(state.normalized_items, dtype=np.float32)
    priors = np.array(evaluation.priors, dtype=np.float32)
    reward = np.array([evaluation.reward], dtype=np.float32)
    episode_data.append((image_data, additional_data, priors, reward))

  pickle.dump(episode_data, file)

def load_episodes(file):
  episodes = []
  file.seek(0)
  while True:
    try:
      episodes.append(pickle.load(file))
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

  image_data = []
  additional_data = []
  for state in states:
    height_map = np.array(state.height_map, dtype=np.float32) / state.bin_height
    image_data.append(np.expand_dims(height_map, axis=0))
    additional_data.append(np.array(state.normalized_items, dtype=np.float32))
  
  image_data = torch.tensor(np.stack(image_data, axis=0), device=device)
  additional_data = torch.tensor(np.stack(additional_data, axis=0), device=device)
  policy, value = model.forward(image_data, additional_data)
  policy = torch.softmax(policy, dim=1)
  result = (policy.cpu().numpy(), value.cpu().numpy())
  return result

def generate_training_data_wrapper(episodes_count):
  global config, model, device

  episodes = generate_episodes(
    config.seed,
    config.seed_pool_size,
    episodes_count,
    config.workers_per_process,
    config.simulations_per_move,
    config.mcts_thread_count,
    config.batch_size,
    config.c_puct,
    config.virtual_loss,
    infer
  )

  return episodes

threshold = None
def generate_training_data(config, model_path, device):
  file = tempfile.TemporaryFile()
  initargs = (config, model_path, device)
  with mp.Pool(config.processes, initializer=init_worker, initargs=initargs) as pool:
    steps_count = (config.episodes_per_iteration + config.step_size - 1) // config.step_size
    args = [config.step_size for _ in range(steps_count)]
    it = pool.imap_unordered(generate_training_data_wrapper, args)
    for episodes in tqdm(it, total=steps_count):
      for episode in episodes:
        dump_episode(episode, file)

  episodes = load_episodes(file)
  rewards = np.array([episode[0][-1][0] for episode in episodes])
  percentile_reward = np.percentile(rewards, config.threshold_percentile)

  global threshold
  if threshold is None:
    threshold = rewards.mean()
  if percentile_reward > threshold:
    momentum = config.threshold_momentum
    threshold = (1 - momentum) * threshold + momentum * percentile_reward

  reshaped_rewards = { +1:0, -1:0 }
  evaluations = []
  for episode in episodes:
    reward = episode[0][-1][0]
    reshaped_reward = +1 if reward > threshold else -1
    reshaped_rewards[reshaped_reward] += 1
    for evaluation in episode:
      evaluation[-1][0] = reshaped_reward

    evaluations.extend(episode)

  data_points_count = len(evaluations)
  wins = reshaped_rewards[+1]
  losses = reshaped_rewards[-1]
  win_ratio = wins / (wins + losses)
  print(f'{data_points_count} data points generated!')
  print(f'Average reward: {rewards.mean():.2f} ± {rewards.std():.3f}')
  print(f'Top {config.threshold_percentile}% reward: {percentile_reward:.2f}')
  print(f'Threshold: {threshold:.3f}')
  print(f'Reshaped reward: {wins} wins, {losses} losses ({win_ratio * 100:.1f}%)')

  return evaluations