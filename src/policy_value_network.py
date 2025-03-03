from bin_packing_solver import State
import torch
from torch import nn

class GlobalPoolingLayer(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x_in):
    avg_vals = torch.mean(x_in, dim=(2, 3))
    max_vals = torch.amax(x_in, dim=(2, 3))
    x_out = torch.cat([avg_vals, max_vals], dim=1)
    return x_out

class GlobalPoolingBiasStructure(nn.Module):
  def __init__(self, c_x, c_g):
    super().__init__()

    self.global_pooling_layer = nn.Sequential(
      nn.BatchNorm2d(c_g),
      nn.ReLU(inplace=True),
      GlobalPoolingLayer(),
      nn.Linear(2 * c_g, c_x)
    )

  def forward(self, x_in, g_in):
    pool_out = self.global_pooling_layer(g_in)
    x_out = x_in + pool_out.view(x_in.shape[0], x_in.shape[1], 1, 1)
    return x_out

class GlobalPoolingResidualBlock(nn.Module):
  def __init__(self, c, c_pool):
    super().__init__()

    self.conv1 = nn.Sequential(
      nn.BatchNorm2d(c),
      nn.ReLU(inplace=True),
      nn.Conv2d(c, c, kernel_size=3, padding=1)
    )

    self.c_pool = c_pool
    self.global_pooling = GlobalPoolingBiasStructure(c - c_pool, c_pool)

    self.conv2 = nn.Sequential(
      nn.BatchNorm2d(c - c_pool),
      nn.ReLU(inplace=True),
      nn.Conv2d(c - c_pool, c, kernel_size=3, padding=1)
    )

  def forward(self, x_in):
    pool_in = self.conv1(x_in)

    pool_x = pool_in[:, self.c_pool:, :, :]
    pool_g = pool_in[:, :self.c_pool, :, :]
    pool_out = self.global_pooling(pool_x, pool_g)

    x_out = self.conv2(pool_out)
    x_out += x_in
    return x_out

class ResidualBlock(nn.Module):
  def __init__(self, c):
    super().__init__()

    self.block = nn.Sequential(
      nn.BatchNorm2d(c),
      nn.ReLU(inplace=True),
      nn.Conv2d(c, c, kernel_size=3, padding=1),
      nn.BatchNorm2d(c),
      nn.ReLU(inplace=True),
      nn.Conv2d(c, c, kernel_size=3, padding=1),
    )

  def forward(self, x_in):
    x_out = self.block(x_in)
    x_out += x_in
    return x_out

class Trunk(nn.Module):
  def __init__(self, c_in, fc_in, n, c, c_pool, pool_count):
    super().__init__()

    self.conv = nn.Conv2d(c_in, c, kernel_size=5, padding=2)
    self.fc = nn.Linear(fc_in, c)

    assert(n % pool_count == 0)
    pool_freq = n // pool_count
    self.residual_blocks = nn.ModuleList([
      ResidualBlock(c)
      if i % pool_freq != pool_freq - 1 
      else GlobalPoolingResidualBlock(c, c_pool)
      for i in range(n)
    ])

    self.bn = nn.BatchNorm2d(c)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, image_in, additional_in):
    image_out = self.conv(image_in)
    additional_out = self.fc(additional_in)
    blocks_in = image_out + additional_out.view(image_out.shape[0], image_out.shape[1], 1, 1)

    blocks_out = blocks_in
    for block in self.residual_blocks:
      blocks_out = block(blocks_out)

    merged_out = self.bn(blocks_out)
    merged_out = self.relu(merged_out)
    return merged_out

class PolicyHead(nn.Module):
  def __init__(self, c, c_head):
    super().__init__()

    self.conv1 = nn.Conv2d(c, c_head, kernel_size=1)
    self.conv2 = nn.Conv2d(c, c_head, kernel_size=1)
    self.global_pooling = GlobalPoolingBiasStructure(c_head, c_head)
    self.final = nn.Sequential(
      nn.BatchNorm2d(c_head),
      nn.ReLU(inplace=True),
      nn.Conv2d(c_head, 1, kernel_size=1),
      nn.Flatten()
    )

  def forward(self, x_in):
    pool_in_x = self.conv1(x_in)
    pool_in_g = self.conv2(x_in)
    pool_out = self.global_pooling(pool_in_x, pool_in_g)
    x_out = self.final(pool_out)
    return x_out

class ValueHead(nn.Module):
  def __init__(self, c, c_head, c_val, c_support):
    super().__init__()

    self.global_pooling = nn.Sequential(
      nn.Conv2d(c, c_head, kernel_size=1),
      GlobalPoolingLayer(),
      nn.Linear(2 * c_head, c_val),
      nn.ReLU(inplace=True),
      nn.Linear(c_val, c_support)
    )

  def forward(self, x_in):
    x_out = self.global_pooling(x_in)
    return x_out

class PolicyValueNetwork(nn.Module):
  def __init__(self, n=6, c=48, c_pool=16, c_head=16, c_val=128, pool_count=2):
    super().__init__()

    self.trunk = Trunk(
      State.input_feature_count,
      State.additional_input_count,
      n, c, c_pool, pool_count
    )

    self.policy_head = PolicyHead(c, c_head)
    self.value_head = ValueHead(c, c_head, c_val, State.value_support_count)

  def forward(self, image_in, additional_in):
    output = self.trunk(image_in, additional_in)
    priors = self.policy_head(output)
    value = self.value_head(output)
    return priors, value