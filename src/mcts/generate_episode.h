#pragma once
#include "mcts.h"
#include "mcts_async.h"
#include <thread>

namespace mcts {

template <typename State>
struct Evaluation {
  Evaluation (const State& _state, int _action_idx = -1, const std::vector<float>& _priors = {}, float _reward = {})
    : state { _state }, action_idx { _action_idx }, priors { _priors }, reward { _reward } {}

  State state;
  int action_idx;
  std::vector<float> priors;
  float reward;
};

template <StateConcept State, InferenceQueueConcept<State> InferenceQueue>
auto generate_episode(
  State state, int simulations_per_move, 
  float c_puct, int virtual_loss, int thread_count, 
  InferenceQueue& inference_queue)
    -> std::vector<Evaluation<State>> {

  using namespace mcts::async;

  std::vector<Evaluation<State>> state_evaluations;
  auto node = std::make_shared<Node<State>>();
  node->state = std::make_unique<State>(state);
  while (true) {
    std::atomic<int> simulation_count;
    auto task = [&] {
      while (simulation_count < simulations_per_move) {
        bool success = run_mcts_simulation<State>(
          node, c_puct, virtual_loss, 
          inference_queue, simulation_count, simulations_per_move
        );
        if (success) ++simulation_count;
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

    std::vector<float> priors(State::action_count);
    int max_visit_count = -1;
    int action_idx;
    NodePtr<State> next_node;
    for (auto child : node->children) {
      if (child->visit_count > max_visit_count) {
        max_visit_count = child->visit_count;
        action_idx = child->action_idx;
        next_node = child;
      }

      priors[child->action_idx] = static_cast<float>(child->visit_count) / (node->visit_count - 1);
    }

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

};