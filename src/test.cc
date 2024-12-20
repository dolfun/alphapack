#include "container.h"
#include "inference_queue.h"
#include "mcts/generate_episode.h"
#include <iostream>
#include <random>
#include <chrono>
#include <format>

int main() {
  std::random_device rd{};
  std::mt19937 engine { rd() };
  std::uniform_int_distribution<int> dist { 4, 8 };

  std::vector<std::string> addresses = { "127.0.0.1:8000" };
  InferenceQueue<Container> inference_queue { 12, addresses };

  int iteration_count = 256;
  for (int i = 0; i < iteration_count; ++i) {
    std::vector<Package> packages(Container::package_count);
    for (auto& pkg : packages) {
      pkg.shape = { dist(engine), dist(engine), dist(engine) };
    }

    int container_height = 24;
    Container container { container_height, packages };

    auto t0 = std::chrono::high_resolution_clock::now();
    auto episode = mcts::generate_episode(container, 256, 5.0, 3, 16, inference_queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    std::cout << std::format("[{}/{}]:{} ({})", i + 1, iteration_count, episode.back().reward, duration) << std::endl;
  }

  std::cout << "ALL DONE!" << std::endl;
}