#pragma once
#include <cmath>
#include <vector>
#include <memory>
#include <cassert>
#include <utility>
#include <algorithm>

/*
  State requirements :-
  Member function: auto possible_actions() const -> std::vector<int>;
  Member function: transition(int action_idx) -> void;
*/

template <typename State>
concept ValidState = requires(State& state, int action_idx) {
  { std::as_const(state).possible_actions() } -> std::same_as<std::vector<int>>;
  { state.transition(action_idx) } -> std::same_as<void>;
};

template <ValidState State>
struct Node {
  Node(const State& _state) : state { _state } {}

  int action_idx = -1;
  State state;

  int visit_count = 0;
  float total_action_value = 0.0f;
  float prior = 0.0f;

  using Ptr = std::shared_ptr<Node<State>>;
  std::vector<Ptr> children;
};

template <ValidState State>
bool run_mcts_simulation(typename Node<State>::Ptr node, auto&& evaluate_state) {
  namespace rng = std::ranges;

  // Selection
  using NodePtr = Node<State>::Ptr;
  std::vector<NodePtr> search_path { node };
  while (node->visit_count > 0) {
    if (node->children.empty()) return false;

    int total_visit_count = node->visit_count - 1;
    node = *rng::max_element(node->children, {}, [total_visit_count] (NodePtr node) {
      constexpr float c_puct = 2.0f;
      float puct_score = c_puct * node->prior * sqrt(total_visit_count) / (1 + node->visit_count);
      float score = node->total_action_value / node->visit_count + puct_score;
      return score;
    });
    search_path.push_back(node);
  }

  // Evaluation
  auto [priors, value] = evaluate_state(node->state);

  // Expansion
  auto actions = node->state.possible_actions();
  float total_valid_prior = 0.0f;
  for (auto action_idx : actions) {
    total_valid_prior += priors[action_idx];
  }

  for (auto action_idx : actions) {
    auto child = std::make_shared<Node<State>>(node->state);
    child->action_idx = action_idx;
    child->state.transition(action_idx);
    child->prior = priors[action_idx] / total_valid_prior;
    node->children.push_back(child);
  }

  // Backpropagation
  for (auto node : search_path) {
    ++node->visit_count;
    node->total_action_value += value;
  }

  return true;
}