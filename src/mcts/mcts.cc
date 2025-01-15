#include "mcts.h"
#include <cmath>
#include <random>
#include <ranges>

namespace mcts {

bool run_mcts_simulation(
    NodePtr node,
    float c_puct,
    int virtual_loss,
    bool is_root,
    float alpha,
    InferenceQueue& inference_queue) {
  // Selection
  std::vector<NodePtr> search_path { node };
  while (node->evaluated) {
    if (node->children.empty()) return true;

    int total_visit_count = node->visit_count - 1;
    auto calculate_score = [c_puct, total_visit_count] (NodePtr node) {
      float mean_action_value = (node->visit_count > 0 ? node->total_action_value / node->visit_count : 0);
      float puct_score = c_puct * node->prior * sqrt(total_visit_count) / (1 + node->visit_count);
      float score = mean_action_value + puct_score;
      return score;
    };

    float max_score = -std::numeric_limits<float>::infinity();
    NodePtr next_node;
    for (auto child : node->children) {
      float score = calculate_score(child);
      if (score > max_score) {
        max_score = score;
        next_node = child;
      }
    }

    node = next_node;
    search_path.push_back(node);
  }

  auto old_visited_value = node->visited.exchange(true);
  if (old_visited_value) return false;
  
  // Apply virtual loss
  for (auto node : search_path) {
    node->visit_count += virtual_loss;
    node->total_action_value -= virtual_loss;
  }

  // Lazy State Update
  if (node->state == nullptr) {
    auto prev_node = node->prev_node.lock();
    node->state = std::make_shared<State>(*prev_node->state);
    node->state->transition(node->action_idx);
  }

  // Enqueue for async evaluation
  auto evaluator_result = inference_queue.infer(node->state);

  // Expansion
  auto actions = node->state->possible_actions();
  for (auto action_idx : actions) {
    auto child = std::make_shared<Node>();
    child->prev_node = node;
    child->action_idx = action_idx;
    node->children.push_back(child);
  }

  // Priors update
  auto [priors, value] = evaluator_result.get();
  float total_valid_prior = 0.0f;
  for (auto action_idx : actions) {
    total_valid_prior += priors[action_idx];
  }

  for (auto child : node->children) {
    child->prior = priors[child->action_idx] / total_valid_prior;

    // Dirichlet Noise
    if (alpha > 0 && is_root) {
      static std::random_device rd{};
      static std::mt19937 engine { rd() };
      std::gamma_distribution<float> dist { alpha };

      size_t k = node->children.size();
      std::vector<float> dirichlet_noise(k);
      std::ranges::generate(dirichlet_noise, [&] { return dist(engine); });
      for (size_t i = 0; i < k; ++i) {
        NodePtr child = node->children[i];
        float noise = dirichlet_noise[i];
        constexpr float epsilon = 0.25f;
        child->prior = (1.0f - epsilon) * child->prior + epsilon * noise;
      }
    }
  }

  // Backpropagation
  for (auto node : search_path) {
    node->visit_count += 1 - virtual_loss;
    node->total_action_value += value + virtual_loss;
  }
  node->evaluated = true;

  return true;
}

}