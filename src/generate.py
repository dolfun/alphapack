from policy_value_network import PolicyValueNetwork
from bin_packing_solver import State
import bin_packing_solver

import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import tempfile
import pickle
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
  model = torch.jit.script(model)
  model.eval()

@torch.no_grad()
def infer(states):
  global model, device

  image_data = [
    np.stack([
      np.array(state.height_map, dtype=np.float32) / State.bin_height,
      np.array(state.feasibility_mask, dtype=np.float32)
    ])
    for state in states
  ]

  additional_data = [
    np.array(state.normalized_items, dtype=np.float32)
    for state in states
  ]

  image_data = torch.tensor(np.stack(image_data), device=device)
  additional_data = torch.tensor(np.stack(additional_data), device=device)
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

def generate_episodes(config, model_path, device, leave_progress_bar=True):
  file = tempfile.TemporaryFile()
  initargs = (config, model_path, device)
  with mp.Pool(config.processes, initializer=init_worker, initargs=initargs) as pool:
    steps_count = (config.episodes_per_iteration + config.step_size - 1) // config.step_size
    args = [config.step_size for _ in range(steps_count)]
    it = pool.imap_unordered(generate_episodes_wrapper, args)
    for episodes in tqdm(it, total=steps_count, leave=leave_progress_bar):
      for episode in episodes:
        pickle.dump(episode, file)

  episodes = load_episodes(file)
  packing_efficiency = np.array([episode[-1].state.packing_efficiency for episode in episodes])
  print(f'Average packing efficiency: {packing_efficiency.mean():.2f} ± {packing_efficiency.std():.3f}')

  move_count = np.array([len(episode) for episode in episodes])
  print(f'{move_count.sum()} moves played! ({move_count.mean():.1f} ± {move_count.std():.1f} moves per game)')

  return episodes