#pragma once
#include <atomic>
#include <memory>
#include <vector>

#include "inference_queue.h"
#include "state.h"

namespace mcts {

struct Node {
  std::shared_ptr<State> state;

  int action_idx { -1 };
  float prior {}, reward {}, init_action_value {};
  std::weak_ptr<Node> prev_node;

  std::atomic<bool> visited {}, evaluated {};
  std::atomic<int> visit_count {};
  std::atomic<float> total_action_value {};

  using Ptr = std::shared_ptr<Node>;
  std::vector<Ptr> children;
};
using NodePtr = Node::Ptr;

enum class SimulationStatus { success, terminal, retry };

SimulationStatus run_mcts_simulation(
  NodePtr root,
  float c_puct,
  int virtual_loss,
  float alpha,
  InferenceQueue& inference_queue
);

}  // namespace mcts