#pragma once
#include "state.h"
#include "inference_queue.h"
#include <atomic>
#include <vector>
#include <memory>

namespace mcts {

struct Node {
  std::shared_ptr<State> state;

  int action_idx{-1};
  float prior{};
  std::weak_ptr<Node> prev_node;

  std::atomic<bool> visited{}, evaluated{};
  std::atomic<int> visit_count{};
  std::atomic<float> total_action_value{};
  std::atomic<float> reward{};

  using Ptr = std::shared_ptr<Node>;
  std::vector<Ptr> children;
};
using NodePtr = Node::Ptr;

bool run_mcts_simulation(
  NodePtr root,
  float c_puct,
  int virtual_loss,
  float alpha,
  InferenceQueue& inference_queue
);

}