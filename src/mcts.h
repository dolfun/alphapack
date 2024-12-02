#pragma once
#include <cmath>
#include <vector>
#include <memory>
#include <cassert>
#include <utility>
#include <algorithm>

template <typename State>
concept StateConcept = requires(State& state, int action_idx) {
  { State::action_count } -> std::convertible_to<size_t>;
  { std::as_const(state).possible_actions() } -> std::same_as<std::vector<int>>;
  { state.transition(action_idx) } -> std::same_as<void>;
  { state.reward() } -> std::same_as<float>;
};

template <typename F, typename State>
concept StateEvaluatorConcept = requires(const State& state, F&& f) {
  { f(state) } -> std::same_as<std::pair<std::vector<float>, float>>;
};

template <StateConcept State>
struct Node {
  template <typename T>
  Node(T&& _state) : state { std::forward<T>(_state) } {}

  State state;
  int action_idx = -1;
  int visit_count = 0;
  float total_action_value = 0.0f;
  float prior = 0.0f;

  using Ptr = std::shared_ptr<Node<State>>;
  std::vector<Ptr> children;
};

template <typename State>
using NodePtr = Node<State>::Ptr;

template <StateConcept State, StateEvaluatorConcept<State> StateEvaluator>
void run_mcts_simulation(NodePtr<State> node, StateEvaluator&& evaluator) {
  namespace rng = std::ranges;

  // Selection
  std::vector<NodePtr<State>> search_path { node };
  while (node->visit_count > 0) {
    if (node->children.empty()) return;

    int total_visit_count = node->visit_count - 1;
    node = *rng::max_element(node->children, {}, [total_visit_count] (NodePtr<State> node) {
      constexpr float c_puct = 2.0f;
      float puct_score = c_puct * node->prior * sqrt(total_visit_count) / (1 + node->visit_count);
      float score = node->total_action_value / node->visit_count + puct_score;
      return score;
    });
    search_path.push_back(node);
  }

  // Evaluation
  auto [priors, value] = evaluator(node->state);

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
}

template <typename State>
struct Evaluation {
  template <typename T>
  Evaluation (T&& _state, int _action_idx = -1, const std::vector<float>& _prior = {}, float _reward = {})
    : state { std::forward<T>(_state) }, action_idx { _action_idx }, prior { _prior }, reward { _reward } {}

  State state;
  int action_idx;
  std::vector<float> prior;
  float reward;
};

template <StateConcept State, StateEvaluatorConcept<State> StateEvaluator>
auto generate_episode(State state, int simulations_per_move, StateEvaluator&& evaluator)
  -> std::vector<Evaluation<State>> {

  std::vector<Evaluation<State>> state_evaluations;
  state_evaluations.emplace_back(state);
  auto node = std::make_shared<Node<State>>(std::move(state));
  while (true) {
    for (int i = 0; i < simulations_per_move; ++i) {
      run_mcts_simulation<State>(node, evaluator);
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

      priors[child->action_idx] = static_cast<float>(child->visit_count) / node->visit_count;
    }

    state_evaluations.emplace_back(std::move(node->state), action_idx, priors);
    node = next_node;
  }

  auto reward = node->state.reward();
  for (auto& evaluation : state_evaluations) {
    evaluation.reward = reward;
  }

  return state_evaluations;
}