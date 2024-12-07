from random import random, randint
from container_solver import Container, Package, Vec3i
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

def normalize_packages(container: Container):
  packages = container.packages
  data = np.zeros(4 * len(packages), dtype=np.float32)
  for i, pkg in enumerate(packages):
    if pkg.is_placed:
      continue

    pkg_x = pkg.shape.x / Container.length
    pkg_y = pkg.shape.y / Container.width
    pkg_z = pkg.shape.z / container.height
    data[4*i : 4*(i+1)] = [
      pkg_x,
      pkg_y,
      pkg_z,
      pkg_x * pkg_y * pkg_z
    ]
  
  return data
