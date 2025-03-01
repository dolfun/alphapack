#include "tree_statistics.h"

namespace mcts {

void compute_tree_statistics_imp(
    NodePtr node,
    int action_idx,
    TreeStatistics& stats,
    int depth = 1) {

  if (node == nullptr) return;

  stats.depths[action_idx] = std::max(stats.depths[action_idx], depth);

  for (auto child : node->children) {
    compute_tree_statistics_imp(child, action_idx, stats, depth + 1);
  }
}

TreeStatistics compute_tree_statistics(NodePtr root) {
  TreeStatistics stats{};

  for (auto child : root->children) {
    stats.init_q_values[child->action_idx] = child->init_action_value;
    if (child->visit_count > 0) {
      stats.final_q_values[child->action_idx] = child->total_action_value / child->visit_count;
    }
    compute_tree_statistics_imp(child, child->action_idx, stats);
  }

  return stats;
}

}