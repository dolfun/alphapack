#include "generate_episode.h"
#include "mcts/generate_episode.h"
#include <random>

auto generate_episode(
  int simulations_per_move, int thread_count, 
  float c_puct, int virtual_loss,
  size_t batch_size, EvaluationQueue::Evaluate_t evaluate)
    -> std::vector<mcts::Evaluation<Container>> {

  static std::random_device rd{};
  static std::mt19937 engine { rd() };
  static std::uniform_int_distribution<int> dist { 4, 8 };

  std::vector<Package> packages(Container::package_count);
  for (auto& package : packages) {
    package.shape = { dist(engine), dist(engine), dist(engine) };
  }

  int container_height = 24;
  Container container { container_height, packages };
  EvaluationQueue evaluation_queue { batch_size, evaluate };
  auto result = mcts::generate_episode(
    container, simulations_per_move, thread_count, c_puct, virtual_loss, evaluation_queue
  );
  return result;
}