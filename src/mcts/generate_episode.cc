#include "generate_episode.h"
#include "mcts.h"
#include <thread>
#include <atomic>
#include <latch>
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

namespace mcts {

auto generate_episode_helper(
  const State& state, int simulations_per_move, int thread_count, 
  float c_puct, int virtual_loss, InferenceQueue& inference_queue)
    -> std::vector<Evaluation> {

  std::vector<Evaluation> state_evaluations;
  auto node = std::make_shared<Node>();
  node->state = std::make_unique<State>(state);
  while (true) {
    std::atomic<int> simulation_count;
    std::latch latch { thread_count + 1 };
    std::atomic<int> finished_count;
    auto task = [&] {
      latch.arrive_and_wait();
      while (simulation_count < simulations_per_move) {
        bool success = run_mcts_simulation(node, c_puct, virtual_loss, inference_queue);
        if (success) ++simulation_count;
      }
      ++finished_count;
    };

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (int i = 0; i < thread_count; ++i) {
      threads.emplace_back(task);
    }

    latch.count_down();
    while (finished_count < thread_count) {
      inference_queue.run();
    }

    for (auto& thread : threads) {
      thread.join();
    }

    if (node->children.empty()) break;

    std::vector<float> priors(State::action_count);
    std::vector<int> weights;
    weights.reserve(node->children.size());
    for (auto child : node->children) {
      weights.push_back(child->visit_count);
      priors[child->action_idx] = static_cast<float>(child->visit_count) / (node->visit_count - 1);
    }

    static std::random_device rd{};
    static std::mt19937 engine { rd() };
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    NodePtr next_node = node->children[dist(engine)];
    int action_idx = next_node->action_idx;

    state_evaluations.emplace_back(*node->state, action_idx, priors);
    next_node->state = std::make_unique<State>(*node->state);
    next_node->state->transition(action_idx);
    node = next_node;
  }

  auto reward = node->state->reward();
  for (auto& evaluation : state_evaluations) {
    evaluation.reward = reward;
  }

  return state_evaluations;
}

auto generate_episode(
  int simulations_per_move, int thread_count, 
  float c_puct, int virtual_loss,
  size_t batch_size, InferenceQueue::InferFunc infer_func)
    -> std::vector<Evaluation> {

  unsigned int seed = get_seed();
  std::mt19937 engine { seed };
  std::uniform_int_distribution<int> dist { 2, 5 };
  std::vector<Item> items(State::item_count);
  for (auto& item : items) {
    item.shape.x = dist(engine);
    item.shape.y = dist(engine);
    item.shape.z = dist(engine);
  }

  State state { items };
  InferenceQueue inference_queue { batch_size, infer_func };
  auto result = generate_episode_helper(
    state, simulations_per_move, thread_count, c_puct, virtual_loss, inference_queue
  );
  return result;
}

};