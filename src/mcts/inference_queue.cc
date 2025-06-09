#include "inference_queue.h"

auto InferenceQueue::infer(Input input) noexcept -> std::future<Result> {
  std::promise<Result> promise;
  auto future = promise.get_future();

  std::unique_lock lock { mutex };
  queue.emplace(input, std::move(promise));
  if (queue.size() >= max_batch_size) {
    lock.unlock();
    cv.notify_one();
  }

  return future;
}

void InferenceQueue::run() noexcept {
  std::unique_lock lock { mutex };
  cv.wait_for(lock, std::chrono::microseconds(100), [&] { return queue.size() >= max_batch_size; });
  size_t batch_count = (queue.size() + max_batch_size - 1) / max_batch_size;
  lock.unlock();

  for (int i = 0; i < batch_count; ++i) {
    lock.lock();

    size_t batch_size = std::min(queue.size(), max_batch_size);
    std::vector<Input> batch(batch_size);
    std::vector<std::promise<Result>> promises(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      batch[i] = queue.front().first;
      promises[i] = std::move(queue.front().second);
      queue.pop();
    }

    lock.unlock();

    auto [priors, values] = infer_func(batch);
    auto priors_ptr = static_cast<float*>(priors.request().ptr);
    auto values_ptr = static_cast<float*>(values.request().ptr);
    for (size_t i = 0; i < batch_size; ++i) {
      auto result = std::make_unique<State::InferResult>();

      std::memcpy(result->priors.data(), priors_ptr, sizeof(float) * State::action_count);
      priors_ptr += State::action_count;
      std::memcpy(result->value.data(), values_ptr, sizeof(float) * State::value_support_count);
      values_ptr += State::value_support_count;

      promises[i].set_value(std::move(result));
    }
  }
}