import torch
from torch import nn
from torch import optim

action_count = 32
try:
  from container_solver import Container
  action_count = Container.action_count
except ImportError:
  pass

class PolicyValueNetwork(nn.Module):
  def __init__(self):
    super(PolicyValueNetwork, self).__init__()

    self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.relu = nn.ReLU()
    self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.flattened_size = 32 * 8 * 8

    self.fc1 = nn.Linear(self.flattened_size + 4 * action_count, 96)
    self.fc2 = nn.Linear(96, 48)
    self.fc3_policy = nn.Linear(48, action_count)
    self.fc3_value = nn.Linear(48, 1)

  def forward(self, height_map, packages_data):
    x = self.cnn1(height_map)
    x = self.relu(x)
    x = self.cnn2(x)
    x = self.relu(x)
    x = self.pool(x)

    x = x.view(x.size(0), -1)
    x = torch.cat((x, packages_data), dim=1)

    x = self.fc1(x)
    x = self.fc2(x)
    policy_output = self.fc3_policy(x)
    value_output = self.fc3_value(x)
    value_output = torch.sigmoid(value_output)

    return policy_output, value_output

def train_policy_value_network(model, dataloader, device):
  model.train()

  learning_rate = 0.05
  epochs_count = 5
  momentum = 0.9

  criterion_policy = nn.CrossEntropyLoss()
  criterion_value = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
  for epoch in range(epochs_count):
    epoch_loss = 0.0

    for inputs in dataloader:
      inputs = (tensor.to(device) for tensor in list(inputs))
      image_data, package_data, priors, reward = inputs

      predicted_priors, predicted_reward = model(image_data, package_data)
      loss = criterion_policy(predicted_priors, priors) + criterion_value(predicted_reward, reward)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs_count}], Loss: {avg_loss:.4f}")