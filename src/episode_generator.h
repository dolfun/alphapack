#pragma once
#include "mcts/generate_episode.h"
#include "container.h"

auto generate_episode(
  const Container& container, int simulations_per_move, 
  float c_puct, int virtual_loss, int thread_count, size_t batch_size, std::vector<std::string> addresses)
    -> std::vector<mcts::Evaluation<Container>>;

float calculate_baseline_reward(Container, std::vector<std::string>);