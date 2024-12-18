#include "episode_generator.h"
#include <iostream>
#include <random>

int main() {
  std::random_device rd{};
  std::mt19937 engine { rd() };
  std::uniform_int_distribution<int> dist { 4, 8 };

  int iteration_count = 256;
  for (int i = 0; i < iteration_count; ++i) {
    std::vector<Package> packages(Container::package_count);
    for (auto& pkg : packages) {
      pkg.shape = { dist(engine), dist(engine), dist(engine) };
    }

    int container_height = 24;
    Container container { container_height, packages };
    std::vector<std::string> addresses = { "127.0.0.1:8000" };
    auto episode = generate_episode(container, 256, 5.0, 3, 16, 12, addresses);

    std::cout << "[" << (i + 1) << "/" << iteration_count << "]:" << episode.back().reward << std::endl;
  }

  std::cout << "ALL DONE!" << std::endl;
}