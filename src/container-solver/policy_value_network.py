import torch
from torch import nn
from torch import optim

action_count = 32
try:
  from container_solver import Container
  action_count = Container.action_count
except ImportError:
  pass

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
    super(ConvBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=False)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = ConvBlock(in_channels, out_channels)
    self.conv2 = ConvBlock(out_channels, out_channels)
    self.relu = nn.ReLU(inplace=False)

  def forward(self, x):
    residual = x
    x = self.conv1(x)
    x = self.conv2(x)
    x = x + residual
    x = self.relu(x)
    return x

class ResidualTower(nn.Module):
  def __init__(self, in_channels, num_residual_blocks):
    super(ResidualTower, self).__init__()
    self.conv_block = ConvBlock(in_channels, 256)
    self.residual_blocks = nn.Sequential(*[ResidualBlock(256, 256) for _ in range(num_residual_blocks)])
    self.conv = nn.Conv2d(256, 1, kernel_size=1)

  def forward(self, x):
    x = self.conv_block(x)
    x = self.residual_blocks(x)
    x = self.conv(x)

    return x


class PolicyHead(nn.Module):
  def __init__(self, input_dims):
    super(PolicyHead, self).__init__()
    self.fc1 = nn.Linear(input_dims, 36)
    self.fc2 = nn.Linear(36, 32)
    self.relu = nn.ReLU(inplace=False)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x


class ValueHead(nn.Module):
  def __init__(self, input_dims):
    super(ValueHead, self).__init__()
    self.fc1 = nn.Linear(input_dims, 36)
    self.fc2 = nn.Linear(36, 18)
    self.fc3 = nn.Linear(18, 1)
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU(inplace=False)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.tanh(x)
    return x

class ResidualNetwork(nn.Module):
  def __init__(self):
    super(ResidualNetwork, self).__init__()
    self.residual_tower = ResidualTower(1, 19)
    self.policy_head = PolicyHead(256)
    self.value_head = ValueHead(256)
    self.conv = nn.Conv2d(256, 1, kernel_size=1)
    self.relu = nn.ReLU(inplace=False)
    self.fc1 = nn.Linear(16 * 16 + 32, 128)
    self.fc2 = nn.Linear(128, 64)

  def forward(self, x):
    x = self.residual_tower(x)
    x = self.conv(x)
    x = self.relu(x)
    x = x.view(x.size(0), -1)

    policy_output = self.policy_head(x)
    value_output = self.value_head(x)
    return policy_output, value_output

class FullyConnectedNetwork(nn.Module):
  def __init__(self, input_size, hidden_layers, output_size):
    super(FullyConnectedNetwork, self).__init__()

    layers = []
    prev_layer_size = input_size

    for hidden_layer_size in hidden_layers:
      layers.append(nn.Linear(prev_layer_size, hidden_layer_size))
      layers.append(nn.ReLU())
      prev_layer_size = hidden_layer_size

    layers.append(nn.Linear(prev_layer_size, output_size))

    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)

class PolicyValueNetwork(nn.Module):
  def __init__(self):
    super(PolicyValueNetwork, self).__init__()
    self.residual_tower = ResidualTower(1, 39)
    self.fully_connected_layer = FullyConnectedNetwork(action_count * 4, [128, 64, 32, 32, 32], 32)
    self.fc1 = nn.Linear(288, 144)
    self.fc2 = nn.Linear(144, 72)
    self.fc3 = nn.Linear(72, 36)
    self.policy_head = PolicyHead(36)
    self.value_head = ValueHead(36)
    self.relu = nn.ReLU(inplace=False)

  def forward(self, image, package_data):
    x = self.residual_tower(image)
    y = self.fully_connected_layer(package_data)
    x = x.view(x.size(0), -1)
    x = torch.cat((x, y), dim=1)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    policy_output = self.policy_head(x)
    value_output = self.value_head(x)
    return policy_output, value_output

def train_policy_value_network(model, dataloader, device):
  model.train()

  learning_rate = 0.005
  epochs_count = 3
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