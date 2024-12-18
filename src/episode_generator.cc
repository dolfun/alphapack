#include "episode_generator.h"
#include "inference_queue.h"

auto generate_episode(
    const Container& container, int simulations_per_move, 
    float c_puct, int virtual_loss, int thread_count, size_t batch_size, std::vector<std::string> addresses)
      -> std::vector<mcts::Evaluation<Container>> {

  InferenceQueue<Container> inference_queue { batch_size, addresses };
  auto episodes = mcts::generate_episode(container, simulations_per_move, c_puct, virtual_loss, thread_count, inference_queue);
  return episodes;
}

float calculate_baseline_reward(Container container, std::vector<std::string> addresses) {
  InferenceQueue<Container> inference_queue { 1, addresses };
  while (true) {
    auto result = inference_queue.enqueue(container);
    auto possible_actions = container.possible_actions();
    auto priors = result.get().first;
    if (possible_actions.empty()) break;
    int max_prior_action = possible_actions[0];
    for (auto action_idx : possible_actions) {
      if (priors[action_idx] > priors[max_prior_action]) {
        max_prior_action = action_idx;
      }
    }
    container.transition(max_prior_action);
  }
  return container.reward();
}