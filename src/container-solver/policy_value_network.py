import torch
from torch import nn
from torch import optim

class ResidualBlock(nn.Module):
  def __init__(self, nr_channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(nr_channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(nr_channels)

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual
    out = self.relu(out)
    return out
  
class PolicyHead(nn.Module):
  def __init__(self, nr_channels, base_size):
    super(PolicyHead, self).__init__()
    self.conv = nn.Conv2d(nr_channels, 2, kernel_size=1, stride=1)
    self.bn = nn.BatchNorm2d(2)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(2 * base_size * base_size, base_size * base_size)

  def forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)
    out = torch.flatten(out, start_dim=1)
    out = self.fc(out)
    return out

class ValueHead(nn.Module):
  def __init__(self, nr_channels, base_size):
    super(ValueHead, self).__init__()
    self.conv = nn.Conv2d(nr_channels, 1, kernel_size=1, stride=1)
    self.bn = nn.BatchNorm2d(1)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(base_size * base_size, 256)
    self.fc2 = nn.Linear(256, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)
    out = torch.flatten(out, start_dim=1)
    out = self.fc1(out)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.sigmoid(out)
    return out

class PolicyValueNetwork(nn.Module):
  def __init__(self, base_size=16, in_channels=1, additional_input_size=128, nr_residual_blocks=10):
    super(PolicyValueNetwork, self).__init__()
    self.base_size = base_size
    self.nr_channels = 256

    self.conv_init = nn.Sequential(
      nn.Conv2d(in_channels, self.nr_channels, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(self.nr_channels),
      nn.ReLU()
    )

    self.residual_blocks = nn.ModuleList([
      ResidualBlock(self.nr_channels) for _ in range(nr_residual_blocks)
    ])

    fc_additional_output_size = 256
    self.fc_additional = nn.Sequential(
      nn.Linear(additional_input_size, 128),
      nn.ReLU(),
      nn.Linear(128, fc_additional_output_size),
      nn.ReLU()
    )

    fc_fusion_input_size = self.nr_channels * self.base_size * self.base_size + fc_additional_output_size
    self.fc_fusion = nn.Sequential(
      nn.Linear(fc_fusion_input_size, self.nr_channels),
      nn.ReLU()
    )

    self.policy_head = PolicyHead(self.nr_channels, base_size)
    self.value_head = ValueHead(self.nr_channels, base_size)

  def forward(self, in_image, in_additional):
    out_image = self.conv_init(in_image)

    for block in self.residual_blocks:
      out_image = block(out_image)

    out_image = torch.flatten(out_image, start_dim=1)
    out_additional = self.fc_additional(in_additional)

    fused = torch.cat((out_image, out_additional), dim=1)
    fused = self.fc_fusion(fused)
    fused = fused.view(-1, self.nr_channels, 1, 1).repeat(1, 1, self.base_size, self.base_size)

    policy = self.policy_head(fused)
    value = self.value_head(fused)

    return policy, value

def train_policy_value_network(model, trainloader, testloader, device):
  model.train()

  learning_rate = 0.005
  epochs_count = 2
  momentum = 0.9

  criterion_policy = nn.CrossEntropyLoss()
  criterion_value = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4)
  for epoch in range(epochs_count):
    epoch_loss = 0.0
    for inputs in trainloader:
      inputs = (tensor.to(device) for tensor in list(inputs))
      image_data, package_data, priors, reward = inputs

      predicted_priors, predicted_reward = model(image_data, package_data)
      loss = criterion_policy(predicted_priors, priors) + criterion_value(predicted_reward, reward)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()

    avg_loss = epoch_loss / len(trainloader)
    print(f'Epoch [{epoch+1}/{epochs_count}], Loss: {avg_loss:.4f}')
    with open('train.csv', 'a') as f:
      f.write(f'{avg_loss},')

  if testloader == None:
    return
  
  model.eval()
  with torch.no_grad():
    total_loss = 0.0
    for inputs in testloader:
      inputs = (tensor.to(device) for tensor in list(inputs))
      image_data, package_data, priors, reward = inputs

      predicted_priors, predicted_reward = model(image_data, package_data)
      loss = criterion_policy(predicted_priors, priors) + criterion_value(predicted_reward, reward)
      total_loss += loss.item()
    avg_loss = total_loss / len(testloader)
    print(f'Loss on test: {avg_loss:.4f}')
    with open('train.csv', 'a') as f:
      f.write(f'{avg_loss}\n')