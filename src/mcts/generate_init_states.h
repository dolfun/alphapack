#pragma once
#include "state.h"

auto generate_random_init_states(uint32_t, size_t, int, int) -> std::vector<State>;
auto generate_cut_init_states(uint32_t, size_t, int, int, float) -> std::vector<State>;