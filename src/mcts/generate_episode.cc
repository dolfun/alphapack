#include "generate_episode.h"
#include "mcts.h"
#include <latch>
#include <ranges>
#include <thread>
#include <atomic>
#include <random>

class SeedPool {
public:
  SeedPool(uint32_t seed, int pool_size) : seeds(pool_size), sample_dist(0, pool_size - 1) {
    std::mt19937 engine { seed };
    std::uniform_int_distribution<uint32_t> seed_dist{};
    std::ranges::generate(seeds, [&] { return seed_dist(engine); });

    static std::random_device rd;
    seed_engine = std::mt19937 { rd() };
  }

  uint32_t get() noexcept {
    return seeds[sample_dist(seed_engine)];
  }

private:
  std::vector<std::uint32_t> seeds;
  std::mt19937 seed_engine;
  std::uniform_int_distribution<size_t> sample_dist;
};

namespace mcts {

auto generate_episode(
  const State& state,
  int move_threshold,
  int simulations_per_move,
  int thread_count,
  float c_puct,
  int virtual_loss,
  InferenceQueue& inference_queue
  ) -> std::vector<Evaluation> {

  std::vector<Evaluation> episode;
  auto node = std::make_shared<Node>();
  node->state = std::make_unique<State>(state);
  while (true) {
    std::atomic<int> simulation_count;
    auto task = [&] {
      while (simulation_count < simulations_per_move) {
        bool success = run_mcts_simulation(node, c_puct, virtual_loss, inference_queue);
        if (success) ++simulation_count;
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (auto _ : std::views::iota(0, thread_count)) {
      threads.emplace_back(task);
    }

    for (auto& thread : threads) {
      thread.join();
    }

    if (node->children.empty()) break;

    std::vector<float> priors(State::action_count);
    int max_visit_count = -1;
    std::vector<int> weights;
    weights.reserve(node->children.size());
    for (auto child : node->children) {
      max_visit_count = std::max(max_visit_count, child->visit_count.load());
      weights.push_back(child->visit_count);
      priors[child->action_idx] = static_cast<float>(child->visit_count) / (node->visit_count - 1);
    }

    if (std::ssize(episode) >= move_threshold) {
      for (auto& weight : weights) {
        weight = (weight == max_visit_count);
      }
    }

    static std::random_device rd{};
    static std::mt19937 engine { rd() };
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    NodePtr next_node = node->children[dist(engine)];
    int action_idx = next_node->action_idx;

    episode.emplace_back(*node->state, action_idx, priors);
    next_node->state = std::make_unique<State>(*node->state);
    next_node->state->transition(action_idx);
    node = next_node;
  }

  auto reward = node->state->reward();
  for (auto& evaluation : episode) {
    evaluation.reward = reward;
  }

  return episode;
}

auto generate_episodes(
  uint32_t seed,
  int seed_pool_size,
  int episodes_count,
  int worker_count,
  int move_threshold,
  int simulations_per_move,
  int mcts_thread_count,
  float c_puct,
  int virtual_loss,
  int batch_size,
  InferenceQueue::InferFunc infer_func
  ) -> std::vector<std::vector<Evaluation>> {

  SeedPool seed_pool { seed, seed_pool_size };
  InferenceQueue inference_queue { static_cast<size_t>(batch_size), infer_func };

  std::mutex episodes_mutex;
  std::vector<std::vector<Evaluation>> episodes;
  auto task = [&] {
    auto seed = seed_pool.get();
    std::mt19937 engine { seed };
    std::uniform_int_distribution<int> dist { 2, 5 };
    std::vector<Item> items(State::item_count);
    for (auto& item : items) {
      item.shape.x = dist(engine);
      item.shape.y = dist(engine);
      item.shape.z = dist(engine);
    }

    State state { items };
    auto episode = generate_episode(
      state,
      move_threshold,
      simulations_per_move,
      mcts_thread_count,
      c_puct,
      virtual_loss,
      inference_queue
    );

    std::lock_guard lock { episodes_mutex };
    episodes.emplace_back(std::move(episode));
  };

  std::atomic<int> work_done, threads_finished;
  std::latch latch { worker_count + 1 };
  auto worker = [&] {
    latch.arrive_and_wait();
    while (true) {
      int current_work = work_done.load();
      if (current_work >= episodes_count) {
        break;
      }

      if (work_done.compare_exchange_strong(current_work, current_work + 1)) {
        task();
      }
    }
    ++threads_finished;
  };

  std::vector<std::thread> threads;
  for (auto _ : std::views::iota(0, worker_count)) {
    threads.emplace_back(worker);
  }

  latch.count_down();
  while (threads_finished < worker_count) {
    inference_queue.run();
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return episodes;
}

};