#pragma once
#include "mcts.h"
#include "container.h"

auto generate_episode(const Container&, int, float, int, int, size_t, std::vector<std::string>)
  -> std::vector<mcts::Evaluation<Container>>;