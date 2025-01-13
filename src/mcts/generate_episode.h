#pragma once
#include "evaluation.h"
#include "inference_queue.h"

namespace mcts {

auto generate_episodes(
  uint32_t seed,
  size_t seed_pool_size,
  int episodes_count,
  int worker_count,
  int simulations_per_move,
  int mcts_thread_count,
  size_t batch_size,
  float c_puct,
  int virtual_loss,
  InferenceQueue::InferFunc infer_func
) -> std::vector<std::vector<Evaluation>>;

};