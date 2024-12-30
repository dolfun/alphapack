#include "generate_episode.h"
#include "mcts/generate_episode.h"
#include <random>

unsigned int get_seed() {
  constexpr int pool_size = 2048;
  static auto seeds = [] {
    std::array<unsigned int, pool_size> seeds{};

    unsigned int seed = 2389473453;
    std::mt19937 engine { seed };
    std::uniform_int_distribution<unsigned int> dist{};
    for (auto& seed : seeds) {
      seed = dist(engine);
    }

    return seeds;
  } ();

  static std::random_device rd{};
  static std::mt19937 seed_engine { rd() };
  static std::uniform_int_distribution<int> seed_dist { 0, pool_size - 1 };
  return seeds[seed_dist(seed_engine)];
}

auto generate_episode(
  int simulations_per_move, int thread_count, 
  float c_puct, int virtual_loss,
  size_t batch_size, EvaluationQueue::Evaluate_t evaluate)
    -> std::vector<mcts::Evaluation<Container>> {

  unsigned int seed = get_seed();
  std::mt19937 engine { seed };
  std::uniform_int_distribution<int> dist { 2, 5 };
  std::vector<Package> packages(Container::package_count);
  for (auto& package : packages) {
    package.shape = { dist(engine), dist(engine), dist(engine) };
  }

  int container_height = 10;
  Container container { container_height, packages };
  EvaluationQueue evaluation_queue { batch_size, evaluate };
  auto result = mcts::generate_episode(
    container, simulations_per_move, thread_count, c_puct, virtual_loss, evaluation_queue
  );
  return result;
}