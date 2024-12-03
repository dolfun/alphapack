import torch
from torch import nn
from unflatten import Info

class PolicyValueNetwork(nn.Module):
  def __init__(self):
    super(PolicyValueNetwork, self).__init__()

    self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.relu = nn.ReLU()
    self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.flattened_size = 32 * 8 * 8

    self.fc1 = nn.Linear(self.flattened_size + 4 * Info.action_count, 64)
    self.fc2_policy = nn.Linear(64, 32)
    self.fc2_value = nn.Linear(64, 1)

  def forward(self, image_data, packages_data):
    x = self.cnn1(image_data)
    x = self.relu(x)
    x = self.cnn2(x)
    x = self.relu(x)
    x = self.pool(x)

    x = x.view(x.size(0), -1)
    x = torch.cat((x, packages_data), dim=1)

    x= self.fc1(x)
    policy_output = self.fc2_policy(x)
    value_output = self.fc2_value(x)

    return policy_output, value_output
  
