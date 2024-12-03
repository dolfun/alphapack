#include <random>
#include <fstream>
#include <cpr/cpr.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include "package_generator.h"
#include "container.h"
#include "../mcts.h"

auto state_evaluator(const Container& container) {
  auto data = container.flatten();

  cpr::Buffer buffer {
    reinterpret_cast<char*>(&data[0]),
    reinterpret_cast<char*>(&data[0] + data.size()),
    "data.bin"
  };
  
  auto response_future = cpr::PostAsync(
    cpr::Url { "0.0.0.0:8000" },
    cpr::Multipart {
      { "type", "inference" },
      { "data", std::move(buffer) }
    }
  );
  const auto& response = response_future.get().text;

  std::vector<float> priors(Container::action_count);
  float value;

  std::memcpy(priors.data(), response.data(), sizeof(float) * priors.size());
  std::memcpy(&value, &(response.end()[-sizeof(float)]), sizeof(float));

  return std::make_pair(priors, value);
}

void generate_episodes(int nr_episodes, int simulations_per_move, std::string output_file) {
  std::random_device rd{};
  std::mt19937 engine { rd() };
  std::uniform_int_distribution<unsigned long> seed_dist;

  std::ofstream file(output_file, std::ios::binary);

  while (nr_episodes--) {
    PackageGenerateInfo generate_info {
      .min_shape_dims = 4, .max_shape_dims = 8,
      .min_cost = 5, .max_cost = 25,
      .priority_percent = 0.25,
      .min_weight_slope = 1.0, .max_weight_slope = 1.0
    };
    auto packages = generate_packages(seed_dist(engine), Container::action_count, generate_info);

    int container_height = 24;
    Container container { container_height, packages, generate_info };
    auto episode = generate_episode(container, simulations_per_move, state_evaluator);
    for (const auto& evaluation : episode) {
      auto data = evaluation.state.flatten();
      data.reserve(data.size() + evaluation.prior.size() + 1);
      std::copy(evaluation.prior.begin(), evaluation.prior.end(), std::back_inserter(data));
      data.push_back(evaluation.reward);

      file.write(reinterpret_cast<const char*>(&data[0]), sizeof(float) * data.size());
    }
  }
}

int main(int argc, const char** argv) {
  assert(argc == 4);

  int nr_episodes = std::stoi(argv[1]);
  int simulations_per_move = std::stoi(argv[2]);
  std::string output_file = std::string(argv[3]);
  generate_episodes(nr_episodes, simulations_per_move, output_file);
}