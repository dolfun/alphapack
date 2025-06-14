from bin_packing_solver import prepare_samples
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ExperienceReplay(Dataset):
    def __init__(self, samples):
        self.samples = [
            (
                np.array(sample.input.image_data, dtype=np.float32),
                np.array(sample.input.additional_data, dtype=np.float32),
                np.array(sample.priors, dtype=np.float32),
                np.array(sample.value, dtype=np.float32),
            )
            for sample in samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def cdf_loss(pred, target):
    pred = F.softmax(pred, dim=1)
    cdf_pred = torch.cumsum(pred, dim=1)
    cdf_target = torch.cumsum(target, dim=1)
    mse_per_sample = torch.sum((cdf_pred - cdf_target) ** 2, dim=1)
    loss = torch.mean(mse_per_sample)
    return loss


step_count = 0


def train_policy_value_network(model, episodes, device, config):
    samples = prepare_samples(episodes)
    experience_replay = ExperienceReplay(samples)
    dataloader = DataLoader(experience_replay, batch_size=2048, shuffle=True)
    print(f"{len(experience_replay)} samples loaded!")

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    train_log_file = open("train_log.csv", "a")
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_priors_loss = 0.0
        epoch_value_pdf_loss = 0.0
        epoch_value_cdf_loss = 0.0
        for inputs in tqdm(dataloader, leave=False):
            inputs = (tensor.to(device) for tensor in list(inputs))
            image_data, additional_data, priors, value = inputs

            pred_priors, pred_value = model(image_data, additional_data)
            pred_priors = F.log_softmax(pred_priors, dim=1)
            pred_value = F.log_softmax(pred_value, dim=1)
            priors_loss = F.kl_div(pred_priors, priors, reduction="batchmean")
            value_pdf_loss = F.kl_div(pred_value, value, reduction="batchmean")
            value_cdf_loss = cdf_loss(pred_value, value)
            total_loss = priors_loss + value_pdf_loss + value_cdf_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_priors_loss += priors_loss.item()
            epoch_value_pdf_loss += value_pdf_loss.item()
            epoch_value_cdf_loss += value_cdf_loss.item()

            global step_count
            step_count += 1
            train_log_file.write(f"{step_count}")
            train_log_file.write(
                f",{total_loss.item()},{priors_loss.item()},{value_pdf_loss.item()},{value_cdf_loss.item()}\n"
            )

        epoch_loss /= len(dataloader)
        epoch_priors_loss /= len(dataloader)
        epoch_value_pdf_loss /= len(dataloader)
        epoch_value_cdf_loss /= len(dataloader)
        print(f"Epoch [{epoch+1}/{config.epochs}] -> ", end="")
        print(
            f"Loss: {epoch_loss:.4f} => {epoch_priors_loss:.4f} + {epoch_value_pdf_loss:.4f} + {epoch_value_cdf_loss:.4f}"
        )
