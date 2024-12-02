#include "package_generator.h"
#include <random>
#include <algorithm>

auto generate_packages(unsigned long seed, int package_count, PackageGenerateInfo info)
  -> std::vector<Package> {

  std::mt19937 engine { seed };
  std::uniform_int_distribution<int> dims_dist { info.min_shape_dims, info.max_shape_dims };
  std::uniform_int_distribution<int> cost_dist { info.min_cost, info.max_cost };
  std::uniform_real_distribution<float> real_dist;

  std::vector<Package> packages(package_count);
  for (auto& package : packages) {
    int shape[3] = { dims_dist(engine), dims_dist(engine), dims_dist(engine) };
    std::ranges::sort(shape);
    package.shape = { shape[0], shape[1], shape[2] };

    float volume = static_cast<float>(package.shape.x) * package.shape.y * package.shape.z;
    package.weight = volume * (info.min_weight_slope
      + real_dist(engine) * (info.max_weight_slope - info.min_weight_slope));

    package.type = (real_dist(engine) < info.priority_percent ? PackageType::priority : PackageType::economy);
    
    package.cost = cost_dist(engine);
  }

  return packages;
}