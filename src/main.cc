#include <fmt/core.h>
#include <algorithm>
#include <random>
#include "container-solver/package_generator.h"
#include "container-solver/container.h"
#include "mcts.h"

int main() {
  std::random_device rd{};
  PackageGenerateInfo generate_info {
    .seed = rd(),
    .package_count = ContainerInfo::max_pkg_cnt,
    .min_shape_dims = 4, .max_shape_dims = 8,
    .min_cost = 5, .max_cost = 25,
    .priority_percent = 0.25,
    .min_weight_slope = 1.0, .max_weight_slope = 1.0
  };
  auto packages = generate_packages(generate_info);

  ContainerInfo container_info {
    .height = 24,
    .weight_limit = 1000,
  };

  ContainerState state { container_info, packages };

  auto evaluate_state = [&] (const ContainerState& state) {
    std::vector<float> policy (ContainerInfo::max_pkg_cnt, 1.0f / ContainerInfo::max_pkg_cnt);
    
    float value = 0.0;
    for (auto pkg : state.get_packages()) {
      if (pkg.is_placed) value += pkg.shape.x * pkg.shape.y * pkg.shape.z;
    }
    value /= container_info.height * ContainerInfo::length * ContainerInfo::width;

    return std::make_pair(policy, value);
  };

  int nr_simulations = 64;
  auto node = std::make_shared<Node<ContainerState>>(state);
  while (true) {
    for (int i = 0; i < nr_simulations; ++i) {
      run_mcts_simulation<ContainerState>(node, evaluate_state);
    }

    if (node->children.empty()) break;
    node = *std::ranges::max_element(node->children, {}, [] (auto node) {
      return node->visit_count;
    });

    fmt::println("Chosen action: {}", node->action_idx);
  }

  float packing_efficiency = 0.0f;
  for (auto pkg : node->state.get_packages()) {
    if (pkg.is_placed) packing_efficiency += pkg.shape.x * pkg.shape.y * pkg.shape.z;
  }
  packing_efficiency /= container_info.height * ContainerInfo::length * ContainerInfo::width;

  fmt::println("Packing Efficiency: {:.2} %", packing_efficiency * 100.0f);
}