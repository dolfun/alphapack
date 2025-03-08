#include "training_sample.h"

void fix_zeros(auto& arr) {
  constexpr float epsilon = 1e-8;
  float sum = 0.0f;
  for (auto& x : arr) {
    x += epsilon;
    sum += x;
  }

  for (auto& x : arr) {
    x /= sum;
  }
}

auto prepare_sample(const mcts::Evaluation& evaluation, int k) -> TrainingSample {
  TrainingSample sample{};
  sample.input = *evaluation.state.inference_input(k);

  Item item = evaluation.state.items().front();
  if (!item.placed) {
    int l = item.shape.x, w = item.shape.y;
    constexpr int L = State::bin_length;
    for (int x = 0; x <= L - l; ++x) {
      for (int y = 0; y <= L - w; ++y) {
        auto [x1, y1] = State::symmetric_transforms[k](x, y, l, w, L);
        int idx = x * L + y;
        int idx1 = x1 * L + y1;
        sample.priors[idx1] = evaluation.priors[idx];
      }
    }
  }

  float scaled_value = evaluation.value * (State::value_support_count - 1);
  int support_idx = static_cast<int>(scaled_value);
  float support_contribution = support_idx + 1 - scaled_value;
  sample.value[support_idx] = support_contribution;
  if (support_idx + 1 < State::value_support_count) {
    sample.value[support_idx + 1] = 1.0f - support_contribution;
  }

  fix_zeros(sample.priors);
  fix_zeros(sample.value);

  return sample;
}

auto prepare_training_samples(const std::vector<std::vector<mcts::Evaluation>>& episodes)
  -> std::vector<TrainingSample> {

  std::vector<TrainingSample> samples;
  for (const auto& episode : episodes) {
    for (const auto& evaluation : episode) {
      for (int k = 0; k < 8; ++k) {
        samples.push_back(prepare_sample(evaluation, k));
      }
    }
  }

  return samples;
}