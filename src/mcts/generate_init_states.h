#pragma once
#include "state.h"

auto generate_random_init_states(uint32_t, int, int, int) -> std::vector<State>;
auto generate_cut_init_states(uint32_t, int, int, int, float, float, int) -> std::vector<State>;