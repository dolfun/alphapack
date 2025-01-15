#pragma once
#include "state.h"
#include "inference_queue.h"
#include <atomic>
#include <vector>
#include <memory>

namespace mcts {

struct Node {
  std::shared_ptr<State> state;
  std::weak_ptr<Node> prev_node;
  
  std::atomic<bool> visited, evaluated;
  std::atomic<int> action_idx = -1;
  std::atomic<int> visit_count;
  std::atomic<float> total_action_value;
  std::atomic<float> prior;

  using Ptr = std::shared_ptr<Node>;
  std::vector<Ptr> children;
};
using NodePtr = Node::Ptr;

bool run_mcts_simulation(
  NodePtr node,
  float c_puct,
  int virtual_loss,
  bool is_root,
  float alpha,
  InferenceQueue& inference_queu
);

}