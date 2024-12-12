#include "episode_generator.h"
#include <cpr/cpr.h>
#include <list>
#include <mutex>
#include <future>
#include <cstring>

using mcts::InferenceResult_t;
template <typename State>
class InferenceQueue {
public:
  InferenceQueue(size_t batch_size, const std::vector<std::string>& _addresses)
  : m_batch_size { batch_size }, addresses { _addresses }, worker_use_count(addresses.size()) {
    daemon = std::jthread(&InferenceQueue<State>::daemon_task, this);
  }

  ~InferenceQueue() {
    to_exit = true;
  }

  InferenceQueue(const InferenceQueue&) = delete;
  InferenceQueue(InferenceQueue&&) = delete;
  const InferenceQueue& operator=(const InferenceQueue&) = delete;
  const InferenceQueue& operator=(InferenceQueue&&) = delete;

  auto enqueue(const State& state) -> std::future<InferenceResult_t> {
    std::lock_guard<std::mutex> lock_guard { mutex };
    std::promise<InferenceResult_t> promise;
    auto future = promise.get_future();
    inference_queue.emplace_back(state.serialize(), std::move(promise));
    return future;
  }

private:
  std::string get_address() noexcept {
    size_t min_idx = 0;
    for (size_t i = 0; i < addresses.size(); ++i) {
      if (worker_use_count[i] < worker_use_count[min_idx]) {
        min_idx = i;
      }
    }
    ++worker_use_count[min_idx];
    return addresses[min_idx];
  }

  void flush() {
    std::unique_lock<std::mutex> lock { mutex, std::defer_lock };
    std::string data;
    std::vector<typename decltype(inference_queue)::iterator> promises;
    lock.lock();
    auto it = inference_queue.begin();
    for (size_t i = 0; i < min(m_batch_size, inference_queue.size()); ++i) {
      data += it->first;
      promises.push_back(it);
      ++it;
    }
    lock.unlock();

    if (promises.empty()) return;

    cpr::Url url { "http://" + get_address() + "/policy_value_inference" };
    cpr::Header header {
      { "batch-size", std::to_string(promises.size()) },
      { "Content-Type", "application/octet-stream" }
    };
    cpr::Body body { &data[0], data.size() };
    auto response = cpr::Post(url, header, body);
    if (response.status_code != 200) return;

    auto text = response.text;
    const char* ptr = &text[0];
    for (auto it : promises) {
      std::vector<float> priors(State::action_count);
      float value;

      std::memcpy(&priors[0], ptr, sizeof(float) * State::action_count);
      ptr += sizeof(float) * State::action_count;
      std::memcpy(&value, ptr, sizeof(float));
      ptr += sizeof(float);

      it->second.set_value(std::make_pair(priors, value));

      lock.lock();
      inference_queue.erase(it);
      lock.unlock();
    }
  }

  void daemon_task() {
    constexpr auto polling_rate = std::chrono::milliseconds(1);
    constexpr auto timeout = std::chrono::milliseconds(75);
    static auto timeout_begin = std::chrono::high_resolution_clock::now();

    while (!to_exit) {
      bool to_flush = false;

      // Timeout
      auto curr_time = std::chrono::high_resolution_clock::now();
      to_flush |= (curr_time - timeout_begin > timeout);

      // Batch filled
      {
        std::lock_guard<std::mutex> lock_guard { mutex };
        to_flush |= (inference_queue.size() >= m_batch_size);
      }

      if (to_flush) {
        flush();
        timeout_begin = curr_time;
      }

      std::this_thread::sleep_for(polling_rate);
    }
  }

  size_t m_batch_size;
  std::vector<std::string> addresses;
  std::vector<size_t> worker_use_count;

  std::mutex mutex;
  std::jthread daemon;
  std::atomic<bool> to_exit;
  std::list<std::pair<std::string, std::promise<InferenceResult_t>>> inference_queue;
};

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