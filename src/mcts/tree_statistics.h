#pragma once
#include "mcts.h"

namespace mcts {

struct TreeStatistics {
  int success_count, terminal_count, retry_count;
  std::array<float, State::action_count> init_q_values, final_q_values;
  std::array<int, State::action_count> depths;
};

TreeStatistics compute_tree_statistics(NodePtr);

};