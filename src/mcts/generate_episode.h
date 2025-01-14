#pragma once
#include "evaluation.h"
#include "inference_queue.h"

namespace mcts {

auto generate_episodes(
  uint32_t seed,
  int seed_pool_size,
  int episodes_count,
  int worker_count,
  int move_threshold,
  int simulations_per_move,
  int mcts_thread_count,
  float c_puct,
  int virtual_loss,
  int batch_size,
  InferenceQueue::InferFunc infer_func
) -> std::vector<std::vector<Evaluation>>;

};