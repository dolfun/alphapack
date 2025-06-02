from bin_packing_solver import generate_cut_init_states, generate_random_init_states
from policy_value_network import PolicyValueNetwork
from config import get_config, get_train_config
from train import train_policy_value_network
from generate import generate_episodes

import argparse
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    try:
        import torch_directml

        device = torch_directml.device()
    except ImportError:
        pass


def get_init_states(config):
    init_states = []
    # init_states += generate_random_init_states(config.seed, 1024, 2, 5)
    init_states += generate_cut_init_states(
        config.seed, config.pool_size, 3, 6, 0.0, 0.0, 1024
    )
    # init_states += generate_cut_init_states(config.seed, config.pool_size, 2, 5, 0.0, 0.75, 1024)
    return init_states


def perform_iteration(checkpoint: dict, checkpoint_path: str, generate_only: bool):
    # Simulate Games
    print("SIMULATING GAMES:")
    checkpoint["iter"] += 1
    config = get_config(-1 if generate_only else checkpoint["iter"])
    init_states = get_init_states(config)
    episodes, packing_efficiency = generate_episodes(
        init_states, config, checkpoint_path, device
    )
    checkpoint["episodes"] = episodes
    checkpoint["packing_efficiency"] = packing_efficiency

    ckpt_id = "gen" if generate_only else checkpoint["iter"]
    torch.save(checkpoint, f"checkpoints/checkpoint_{ckpt_id}.ckpt")
    del checkpoint["episodes"]
    del checkpoint["packing_efficiency"]
    print()

    if generate_only:
        return

    # Train
    print("TRAINING:")
    model = PolicyValueNetwork().to(device)
    model.load_state_dict(checkpoint["model"])
    train_config = get_train_config(checkpoint["iter"])
    train_policy_value_network(model, episodes, device, train_config)
    checkpoint["model"] = model.state_dict()
    torch.save(checkpoint, checkpoint_path)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_count", type=int, default=1)
    parser.add_argument("--generate_only", action="store_true")
    args = parser.parse_args()

    checkpoint_path = "policy_value_network.ckpt"
    if not os.path.exists(checkpoint_path):
        print("Creating new model!")
        model = PolicyValueNetwork().to(device)
        checkpoint = {
            "iter": 0,
            "model": model.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

    os.makedirs("checkpoints", exist_ok=True)
    for _ in range(args.iteration_count):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"[{checkpoint['iter'] + 1}]")
        perform_iteration(checkpoint, checkpoint_path, args.generate_only)


if __name__ == "__main__":
    main()
