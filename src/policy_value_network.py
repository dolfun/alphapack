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

class PolicyValueNetwork(nn.Module):
  def __init__(self, base_size=16, in_channels=1, additional_input_size=128, nr_residual_blocks=10):
    super(PolicyValueNetwork, self).__init__()
    nr_channels = 256

    self.conv_init = nn.Sequential(
      nn.Conv2d(in_channels, nr_channels, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(nr_channels),
      nn.ReLU()
    )

    self.residual_blocks = nn.ModuleList([
      ResidualBlock(nr_channels) for _ in range(nr_residual_blocks)
    ])

    conv_final_nr_channels = 2
    self.conv_final = nn.Sequential(
      nn.Conv2d(nr_channels, conv_final_nr_channels, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(conv_final_nr_channels),
      nn.ReLU()
    )

    fc_additional_output_size = 64
    self.fc_additional = nn.Sequential(
      nn.Linear(additional_input_size, 128),
      nn.ReLU(),
      nn.Linear(128, fc_additional_output_size),
      nn.ReLU()
    )

    base_area = base_size * base_size
    fc_fusion_input_size = conv_final_nr_channels * base_area + fc_additional_output_size
    fc_fusion_output_size = 256
    self.fc_fusion = nn.Sequential(
      nn.Linear(fc_fusion_input_size, 384),
      nn.ReLU(),
      nn.Linear(384, fc_fusion_output_size),
      nn.ReLU()
    )

    self.policy_head = nn.Linear(fc_fusion_output_size, base_area)

    self.value_head = nn.Sequential(
      nn.Linear(fc_fusion_output_size, 256),
      nn.ReLU(),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

  def forward(self, in_image, in_additional):
    out_image = self.conv_init(in_image)

    for block in self.residual_blocks:
      out_image = block(out_image)

    out_image = self.conv_final(out_image)
    
    out_image = torch.flatten(out_image, start_dim=1)
    out_additional = self.fc_additional(in_additional)
    fused = torch.cat((out_image, out_additional), dim=1)
    fused = self.fc_fusion(fused)

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