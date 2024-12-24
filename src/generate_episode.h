#pragma once
#include "mcts/mcts_common.h"
#include "evaluation_queue.h"

auto generate_episode(int, int, float, int, size_t, EvaluationQueue::Evaluate_t)
  -> std::vector<mcts::Evaluation<Container>>;