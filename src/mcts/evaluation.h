#pragma once
#include "state.h"
#include "tree_statistics.h"

namespace mcts {

struct Evaluation {
  State state;
  int action_idx;
  std::array<float, State::action_count> priors;
  float value;
  TreeStatistics tree_statistics;
};

}