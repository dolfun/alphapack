#include "mcts.h"
#include <cmath>
#include <random>
#include <ranges>
#include <algorithm>

namespace mcts {

bool run_mcts_simulation(
    NodePtr root,
    float c_puct,
    int virtual_loss,
    float alpha,
    InferenceQueue& inference_queue) {

  // Selection
  std::vector<NodePtr> search_path { root };
  while (true) {
    auto current_node = search_path.back();
    if (!current_node->evaluated.load(std::memory_order_acquire)) break;

    // Terminal State Condition
    if (current_node->children.empty()) {
      float cumulative_reward = 0.0f;
      for (auto node : std::views::reverse(search_path)) {
        node->visit_count.fetch_add(1, std::memory_order_relaxed);
        node->total_action_value.fetch_add(cumulative_reward, std::memory_order_relaxed);
        cumulative_reward += node->reward;
      }
      return true;
    }

    float sqrt_parent_visit_count = std::sqrt(current_node->visit_count.load(std::memory_order_relaxed));
    auto calculate_ucb_score = [c_puct, sqrt_parent_visit_count] (NodePtr child) {
      int visit_count = child->visit_count.load(std::memory_order_relaxed);
      float total_action_value = child->total_action_value.load(std::memory_order_relaxed);

      float value_score = (visit_count > 0 ? child->reward + total_action_value / visit_count : 0);
      float prior_score = c_puct * child->prior * sqrt_parent_visit_count / (1 + visit_count);
      return value_score + prior_score;
    };

    NodePtr next_node = *std::ranges::max_element(current_node->children, {}, calculate_ucb_score);
    search_path.push_back(next_node);
  }

  auto leaf_node = search_path.back();
  auto old_visited_value = leaf_node->visited.exchange(true, std::memory_order_seq_cst);
  if (old_visited_value) return false;
  
  // Apply virtual loss
  for (auto node : search_path) {
    node->visit_count.fetch_add(virtual_loss, std::memory_order_relaxed);
    node->total_action_value.fetch_sub(virtual_loss, std::memory_order_relaxed);
  }

  // Lazy State Update
  if (leaf_node->state == nullptr) {
    auto leaf_parent_node = leaf_node->prev_node.lock();
    leaf_node->state = std::make_shared<State>(*leaf_parent_node->state);
    leaf_node->reward = leaf_node->state->transition(leaf_node->action_idx);
  }

  // Enqueue for async evaluation
  thread_local static std::random_device rd{};
  thread_local static std::mt19937 engine { rd() };
  thread_local static std::uniform_int_distribution<int> symmetry_dist(0, 7);

  int symmetry_idx = symmetry_dist(engine);
  auto transformed_state = get_state_symmetry(*leaf_node->state, symmetry_idx);
  auto inference_result = inference_queue.infer(std::make_shared<State>(std::move(transformed_state)));

  // Expansion
  auto actions = leaf_node->state->possible_actions();
  leaf_node->children.reserve(actions.size());
  for (auto action_idx : actions) {
    auto child = std::make_shared<Node>();
    child->prev_node = leaf_node;
    child->action_idx = action_idx;
    leaf_node->children.push_back(child);
  }

  // Waiting for evaluation result
  auto [transformed_priors, value_supports] = inference_result.get();

  // Applying softmax
  auto apply_softmax = [] (auto& arr) {
    float sum = 0.0f;
    float max_val = *std::ranges::max_element(arr);
    for (auto& val : arr) {
      val = std::exp(val - max_val);
      sum += val;
    }

    for (auto& val : arr) {
      val /= sum;
    }
  };

  apply_softmax(transformed_priors);
  apply_softmax(value_supports);

  // Extract value from supports
  float value = 0.0f;
  for (int i = 0; i < State::value_support_count; ++i) {
    value += value_supports[i] * i / (State::value_support_count - 1); 
  }

  // Apply innverse dihedral transform
  auto priors = get_inverse_priors_symmetry(*leaf_node->state, transformed_priors, symmetry_idx);

  float total_valid_prior = 0.0f;
  for (auto action_idx : actions) {
    total_valid_prior += priors[action_idx];
  }

  for (auto child : leaf_node->children) {
    child->prior = priors[child->action_idx] / total_valid_prior;
  }

  // Dirichlet Noise
  if (alpha > 0.0f && !leaf_node->children.empty() && leaf_node == root) {
    std::gamma_distribution<float> dist { alpha };
    std::vector<float> dirichlet_noise(leaf_node->children.size());
    std::ranges::generate(dirichlet_noise, [&] { return dist(engine); });
    auto dirichlet_noise_sum = std::reduce(dirichlet_noise.begin(), dirichlet_noise.end());

    for (size_t i = 0; i < leaf_node->children.size(); ++i) {
      NodePtr child = leaf_node->children[i];
      float noise = dirichlet_noise[i] / dirichlet_noise_sum;
      constexpr float epsilon = 0.25f;
      child->prior = (1.0f - epsilon) * child->prior + epsilon * noise;
    }
  }

  // Backup
  float cumulative_reward = value;
  for (auto node : std::views::reverse(search_path)) {
    node->visit_count.fetch_add(1 - virtual_loss, std::memory_order_relaxed);
    node->total_action_value.fetch_add(cumulative_reward + virtual_loss, std::memory_order_relaxed);
    cumulative_reward += node->reward;
  }
  leaf_node->evaluated.store(true, std::memory_order_release);

  return true;
}

}