from random import random, randint
from container_solver import Package, Vec3i
import numpy as np

class GenerateInfo:
  dims_range = (4, 8)
  weight_range = (5, 10)
  priority_ratio = 0.25
  cost_range = (5, 50)

def random_package():
  pkg = Package()
  shape = sorted(randint(*GenerateInfo.dims_range) for i in range(3))
  pkg.shape = Vec3i(shape[0], shape[1], shape[2])
  pkg.weight = randint(*GenerateInfo.weight_range)
  pkg.is_priority = True if random() < GenerateInfo.priority_ratio else False
  pkg.cost = randint(*GenerateInfo.cost_range)
  return pkg

def normalize_packages(packages: list[Package]):
  def normalize(val, range):
    min, max = range
    return (val - min) / (max - min)

  data = np.zeros(4 * len(packages), dtype=np.float32)
  for i, pkg in enumerate(packages):
    if pkg.is_placed:
      continue

    data[4*i : 4*(i+1)] = [
      normalize(pkg.shape.x, GenerateInfo.dims_range),
      normalize(pkg.shape.y, GenerateInfo.dims_range),
      normalize(pkg.shape.z, GenerateInfo.dims_range),
      normalize(pkg.cost, GenerateInfo.cost_range),
    ]
  
  return data
