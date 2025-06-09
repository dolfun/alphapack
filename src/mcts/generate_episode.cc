#include "generate_episode.h"

#include <atomic>
#include <latch>
#include <random>
#include <ranges>
#include <thread>

#include "mcts.h"
#include "tree_statistics.h"

namespace mcts {

auto generate_episode(
  const State& state,
  int move_threshold,
  int simulations_per_move,
  int thread_count,
  float c_puct,
  int virtual_loss,
  float alpha,
  InferenceQueue& inference_queue
) -> std::vector<Evaluation> {
  std::vector<Evaluation> episode;
  auto node = std::make_shared<Node>();
  node->state = std::make_shared<State>(state);
  while (true) {
    // Inject Dirichlet Noise in root node once
    if (alpha > 0.0f) {
      run_mcts_simulation(node, c_puct, virtual_loss, alpha, inference_queue);
    }

    std::atomic<int> simulation_count {};
    std::atomic<int> success_count {}, terminal_count {}, retry_count {};
    auto task = [&] {
      while (simulation_count < simulations_per_move) {
        auto status = run_mcts_simulation(node, c_puct, virtual_loss, -1.0f, inference_queue);

        switch (status) {
          case SimulationStatus::success:
            ++success_count;
            ++simulation_count;
            break;

          case SimulationStatus::terminal:
            ++terminal_count;
            ++simulation_count;
            break;

          case SimulationStatus::retry:
            ++retry_count;
            std::this_thread::sleep_for(std::chrono::microseconds(100));  // !?
            break;
        };
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (int i = 0; i < thread_count; ++i) {
      threads.emplace_back(task);
    }

    for (auto& thread : threads) {
      thread.join();
    }

    if (node->children.empty()) break;

    std::array<float, State::action_count> priors {};
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

    thread_local static std::random_device rd {};
    thread_local static std::mt19937 engine { rd() };
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    NodePtr next_node = node->children[dist(engine)];
    if (next_node->state == nullptr) {
      next_node->state = std::make_shared<State>(*node->state);
      next_node->reward = next_node->state->transition(next_node->action_idx);
    }

    auto tree_statistics = compute_tree_statistics(node);
    tree_statistics.success_count = success_count;
    tree_statistics.terminal_count = terminal_count;
    tree_statistics.retry_count = retry_count;

    episode.emplace_back(
      *node->state,
      next_node->action_idx,
      priors,
      next_node->reward,
      tree_statistics
    );
    node = next_node;
  }

  episode.emplace_back(
    *node->state,
    -1,
    std::array<float, State::action_count> {},
    0.0f,
    TreeStatistics {}
  );

  float cumulative_reward = 0.0f;
  for (auto& evaluation : std::views::reverse(episode)) {
    cumulative_reward += evaluation.value;
    evaluation.value = cumulative_reward;
  }

  return episode;
}

auto generate_episodes(
  const std::vector<State>& states,
  int episodes_count,
  int worker_count,
  int move_threshold,
  int simulations_per_move,
  int mcts_thread_count,
  float c_puct,
  int virtual_loss,
  float alpha,
  int batch_size,
  InferenceQueue::InferFunc infer_func
) -> std::vector<std::vector<Evaluation>> {
  std::mutex episodes_mutex;
  std::vector<std::vector<Evaluation>> episodes;
  episodes.reserve(episodes_count);

  std::random_device rd {};
  std::mt19937 engine { rd() };
  std::uniform_int_distribution<size_t> dist { 0, states.size() - 1 };

  std::mutex rng_mutex;
  auto sample_state_idx = [&] {
    std::lock_guard lock { rng_mutex };
    return dist(engine);
  };

  InferenceQueue inference_queue { static_cast<size_t>(batch_size), infer_func };
  std::atomic<int> work_done {}, threads_finished {};
  std::latch latch { worker_count + 1 };
  auto worker = [&] {
    latch.arrive_and_wait();

    while (work_done.fetch_add(1) < episodes_count) {
      int state_idx = sample_state_idx();
      auto episode = generate_episode(
        states[state_idx],
        move_threshold,
        simulations_per_move,
        mcts_thread_count,
        c_puct,
        virtual_loss,
        alpha,
        inference_queue
      );

      std::lock_guard lock { episodes_mutex };
      episodes.emplace_back(std::move(episode));
    }

    ++threads_finished;
  };

  std::vector<std::thread> threads;
  threads.reserve(worker_count);
  for (int i = 0; i < worker_count; ++i) {
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

};  // namespace mcts