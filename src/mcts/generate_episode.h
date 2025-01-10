#pragma once
#include "evaluation.h"
#include "inference_queue.h"

namespace mcts {

auto generate_episode(
  int simulations_per_move, int thread_count, 
  float c_puct, int virtual_loss,
  size_t batch_size, InferenceQueue::InferFunc infer_func)
    -> std::vector<Evaluation>;

};