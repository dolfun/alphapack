#include <fmt/core.h>
#include <random>
#include "package_generator.h"
#include "container.h"
#include "../mcts.h"

int main() {
  std::random_device rd{};
  PackageGenerateInfo generate_info {
    .seed = rd(),
    .package_count = Container::action_count,
    .min_shape_dims = 4, .max_shape_dims = 8,
    .min_cost = 5, .max_cost = 25,
    .priority_percent = 0.25,
    .min_weight_slope = 1.0, .max_weight_slope = 1.0
  };
  auto packages = generate_packages(generate_info);

  Container container { 24, packages };

  auto evaluator = [&] (const Container& container) {
    std::vector<float> priors (Container::action_count, 1.0f / Container::action_count);
    
    float value = 0.0f;
    for (auto pkg : container.packages()) {
      if (pkg.is_placed) value += pkg.shape.x * pkg.shape.y * pkg.shape.z;
    }
    value /= container.height() * Container::length * Container::width;

    return std::make_pair(priors, value);
  };

  int simulations_per_move = 64;
  auto episode = generate_episode(container, simulations_per_move, evaluator);
  for (const auto& evaluation : episode) {
    fmt::println("Chosen Action: {}", evaluation.action_idx);
  }
  fmt::println("Reward: {:.2} %", episode.back().reward * 100.0f);
}