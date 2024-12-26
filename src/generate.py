from container_solver import generate_episode
from tqdm import tqdm
import pickle
import numpy as np
import torch

def save_evaluation(evaluation, file):
  container = evaluation.container
  height_map = np.array(container.height_map, dtype=np.float32) / container.height
  image_data = np.expand_dims(height_map, axis=0)
  additional_data = np.array(container.normalized_packages, dtype=np.float32)
  priors = np.array(evaluation.priors, dtype=np.float32)
  reward = np.array([evaluation.reward], dtype=np.float32)
  pickle.dump((image_data, additional_data, priors, reward), file)

@torch.no_grad()
def evaluate_with_model(model, device, containers):
  image_data = []
  additional_data = []
  for container in containers:
    height_map = np.array(container.height_map, dtype=np.float32) / container.height
    image_data.append(np.expand_dims(height_map, axis=0))
    additional_data.append(np.array(container.normalized_packages, dtype=np.float32))
  
  image_data = torch.tensor(np.stack(image_data, axis=0), device=device)
  additional_data = torch.tensor(np.stack(additional_data, axis=0), device=device)
  policy, value = model.forward(image_data, additional_data)
  policy = torch.softmax(policy, dim=1)
  result = (policy.cpu().numpy(), value.cpu().numpy())
  return result

def generate_training_data(config, model, device, episodes_file):
  model.eval()
  evaluate = lambda containers: evaluate_with_model(model, device, containers)

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
      save_evaluation(evaluation, episodes_file)

  rewards = np.array(rewards)
  print(f'{data_points_count} data points generated')
  print(f'Average reward: {rewards.mean():.2} ± {rewards.std():.3f}')