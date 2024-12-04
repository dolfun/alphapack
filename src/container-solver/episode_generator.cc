#include "episode_generator.h"
#include <cpr/cpr.h>
#include <cstring>

auto state_evaluator(const Container& container) -> std::pair<std::vector<float>, float> {
  auto data = container.serialize();
  auto post_result = cpr::PostAsync(
    cpr::Url { "http://127.0.0.1:8000/policy_value_inference" },
    cpr::Header {{ "Content-Type", "application/octet-stream" }},
    cpr::Body { &data[0], data.size() }
  );

  std::vector<float> priors(Container::action_count);
  float value;

  const auto& result = post_result.get().text;
  std::memcpy(&priors[0], &result[0], sizeof(float) * Container::action_count);
  std::memcpy(&value, &(result.end()[-sizeof(float)]), sizeof(float));
  return std::make_pair(priors, value);
};

auto generate_episode(const Container& container, int simulations_per_move, float c_puct)
    -> std::vector<mcts::Evaluation<Container>> {
  auto episodes = mcts::generate_episode(container, simulations_per_move, c_puct, state_evaluator);
  return episodes;
}