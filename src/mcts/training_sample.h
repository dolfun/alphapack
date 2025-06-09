#pragma once
#include "evaluation.h"
#include "state.h"

struct TrainingSample {
  State::InferInput input;
  std::array<float, State::action_count> priors {};
  std::array<float, State::value_support_count> value {};
};

auto prepare_training_samples(const std::vector<std::vector<mcts::Evaluation>>& episodes)
  -> std::vector<TrainingSample>;