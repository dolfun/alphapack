#include "episode_generator.h"
#include <cpr/cpr.h>
#include <cstring>
#include <thread>
#include <queue>
#include <mutex>
#include <future>

using mcts::InferenceResult_t;
template <typename State>
class InferenceQueue {
public:
  InferenceQueue(size_t batch_size) : m_batch_size { batch_size } {}

  size_t batch_size() const noexcept { return m_batch_size; }

  auto enqueue(const State& state) -> std::future<InferenceResult_t> {
    std::lock_guard<std::mutex> lock_guard { mutex };
    std::promise<InferenceResult_t> promise;
    auto future = promise.get_future();
    inference_queue.emplace(state, std::move(promise));
    if (inference_queue.size() >= m_batch_size) {
      auto thread = std::thread(&InferenceQueue::flush, this);
      thread.detach();
    }

    return future;
  }

  void flush() {
    std::string data;
    std::vector<std::promise<InferenceResult_t>> promises;
    {
      std::lock_guard<std::mutex> lock_guard { mutex };
      if (inference_queue.empty()) return;
      for (size_t i = 0; i < m_batch_size && !inference_queue.empty(); ++i) {
        auto [state, promise] = std::move(inference_queue.front());
        inference_queue.pop();

        data += state.serialize();
        promises.emplace_back(std::move(promise));
      }
    }

    auto post_result = cpr::PostAsync(
      cpr::Url { "http://127.0.0.1:8000/policy_value_inference" },
      cpr::Header {
        { "batch-size", std::to_string(promises.size()) },
        { "Content-Type", "application/octet-stream" }
      },
      cpr::Body { &data[0], data.size() }
    );

    auto text = post_result.get().text;
    const char* ptr = &text[0];
    for (auto& promise : promises) {
      std::vector<float> priors(State::action_count);
      float value;

      std::memcpy(&priors[0], ptr, sizeof(float) * State::action_count);
      ptr += sizeof(float) * State::action_count;
      std::memcpy(&value, ptr, sizeof(float));
      ptr += sizeof(float);

      promise.set_value(std::make_pair(priors, value));
    }
    return;
  }

private:
  std::mutex mutex;
  size_t m_batch_size;
  std::queue<std::pair<State, std::promise<InferenceResult_t>>> inference_queue;
};

auto generate_episode(
    const Container& container, int simulations_per_move, 
    float c_puct, int virtual_loss, int thread_count, size_t batch_size)
      -> std::vector<mcts::Evaluation<Container>> {

  InferenceQueue<Container> inference_queue { batch_size };
  auto episodes = mcts::generate_episode(container, simulations_per_move, c_puct, virtual_loss, thread_count, inference_queue);
  return episodes;
}