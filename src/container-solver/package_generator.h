#pragma once
#include "package.h"
#include <vector>

struct PackageGenerateInfo {
  unsigned long seed;
  int package_count;
  int min_shape_dims, max_shape_dims;
  int min_cost, max_cost;
  float priority_percent;
  float min_weight_slope, max_weight_slope;
};

auto generate_packages(PackageGenerateInfo) -> std::vector<Package>;