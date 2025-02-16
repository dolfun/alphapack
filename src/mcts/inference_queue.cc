#include "inference_queue.h"

auto InferenceQueue::infer(std::shared_ptr<State> state) noexcept -> std::future<Result> {
  std::promise<Result> promise;
  auto future = promise.get_future();

  std::unique_lock lock { mutex };
  queue.emplace(state, std::move(promise));
  if (queue.size() >= max_batch_size) {
    lock.unlock();
    cv.notify_one();
  }

  return future;
}

void InferenceQueue::run() noexcept {
  std::unique_lock lock { mutex };
  cv.wait_for(lock, std::chrono::microseconds(100), [&] {
    return queue.size() >= max_batch_size;
  });
  size_t batch_count = (queue.size() + max_batch_size - 1) / max_batch_size;
  lock.unlock();

  for (int i = 0; i < batch_count; ++i) {
    lock.lock();
    size_t batch_size = std::min(queue.size(), max_batch_size);
    std::vector<std::shared_ptr<State>> states(batch_size);
    std::vector<std::promise<Result>> promises(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      states[i] = queue.front().first;
      promises[i] = std::move(queue.front().second);
      queue.pop();
    }
    lock.unlock();

    auto [priors, values] = infer_func(states);
    auto priors_ptr = static_cast<float*>(priors.request().ptr);
    auto values_ptr = static_cast<float*>(values.request().ptr);
    for (size_t i = 0; i < batch_size; ++i) {
      Result result;
      result.first.resize(State::action_count);
      std::memcpy(result.first.data(), priors_ptr, sizeof(float) * State::action_count);
      priors_ptr += State::action_count;

      result.second = values_ptr[i];
      promises[i].set_value(std::move(result));
    }
  }
}