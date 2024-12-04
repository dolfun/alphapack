#pragma once
#include "../mcts.h"
#include "container.h"

auto generate_episode(const Container&, int, float)
  -> std::vector<mcts::Evaluation<Container>>;