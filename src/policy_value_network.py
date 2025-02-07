from bin_packing_solver import State
from torch import nn

class ResidualBlock(nn.Module):
  def __init__(self, nr_channels):
    super().__init__()

    self.block = nn.Sequential(
      nn.BatchNorm2d(nr_channels),
      nn.ReLU(),
      nn.Conv2d(nr_channels, nr_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(nr_channels),
      nn.ReLU(),
      nn.Conv2d(nr_channels, nr_channels, kernel_size=3, padding=1),
    )

  def forward(self, x_in):
    x_out = self.block(x_in)
    x_out += x_in
    return x_out

class Trunk(nn.Module):
  def __init__(self, nr_input_features, additional_input_size, nr_residual_blocks, nr_channels):
    super().__init__()

    self.conv = nn.Conv2d(nr_input_features, nr_channels, kernel_size=5, padding=2)
    self.fc = nn.Linear(additional_input_size, nr_channels)
    self.residual_blocks = nn.ModuleList([
      ResidualBlock(nr_channels) for _ in range(nr_residual_blocks)
    ])
    self.bn = nn.BatchNorm2d(nr_channels)
    self.relu = nn.ReLU()

  def forward(self, image_in, additional_in):
    image_out = self.conv(image_in)
    additional_out = self.fc(additional_in)
    merged_in = image_out + additional_out.view(image_out.shape[0], image_out.shape[1], 1, 1)
    for block in self.residual_blocks:
      merged_out = block(merged_in)

    merged_out = self.bn(merged_out)
    merged_out = self.relu(merged_out)
    return merged_out
  
class PolicyHead(nn.Module):
  def __init__(self, nr_channels, image_size):
    super().__init__()

    base_size = image_size * image_size
    self.policy_head = nn.Sequential(
      nn.Conv2d(nr_channels, 2, kernel_size=1),
      nn.BatchNorm2d(2),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(2 * base_size, base_size)
    )

  def forward(self, x_in):
    x_out = self.policy_head(x_in)
    return x_out

class ValueHead(nn.Module):
  def __init__(self, nr_channels, image_size):
    super().__init__()

    base_size = image_size * image_size
    self.value_head = nn.Sequential(
      nn.Conv2d(nr_channels, 2, kernel_size=1),
      nn.BatchNorm2d(2),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(2 * base_size, 256),
      nn.ReLU(),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

  def forward(self, x_in):
    x_out = self.value_head(x_in)
    return x_out

class PolicyValueNetwork(nn.Module):
  def __init__(self, nr_input_features=2, nr_residual_blocks=6, nr_channels=32):
    super().__init__()

    additional_input_size = State.item_count * State.values_per_item
    self.trunk = Trunk(nr_input_features, additional_input_size, nr_residual_blocks, nr_channels)
    self.policy_head = PolicyHead(nr_channels, image_size=State.bin_length)
    self.value_head = ValueHead(nr_channels, image_size=State.bin_length)

  def forward(self, image_in, additional_in):
    output = self.trunk(image_in, additional_in)
    priors = self.policy_head(output)
    value = self.value_head(output)
    return priors, value