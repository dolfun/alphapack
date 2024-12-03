import pickle
import argparse
from container_solver import Container, generate_episode
from package_utils import random_package

args = None

def generate_training_data():
  for _ in range(args.games_per_epoch):
    packages = [random_package() for _ in range(Container.action_count)]

    container_height = 24
    container = Container(container_height, packages)

    episodes = generate_episode(container, args.simulations_per_move)
    with open(args.episodes_output, 'wb') as f:
      for episode in episodes:
        pickle.dump((episode.container, episode.action_idx, episode.prior, episode.reward), f)

def epoch():
  generate_training_data()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch_count', type=int, default=1)
  parser.add_argument('--games_per_epoch', type=int, default=1)
  parser.add_argument('--simulations_per_move', type=int, default=16)
  parser.add_argument('--episodes_output', default='episodes.bin')

  global args
  args = parser.parse_args()

  for _ in range(args.epoch_count):
    epoch()

if __name__ == '__main__':
  main()
