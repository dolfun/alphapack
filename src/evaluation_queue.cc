#include "evaluation_queue.h"

auto EvaluationQueue::enqueue(std::shared_ptr<Container> container) noexcept -> std::future<Result_t> {
  std::promise<Result_t> promise;
  auto future = promise.get_future();

  std::unique_lock lock { mutex };
  queue.emplace(container, std::move(promise));
  if (queue.size() >= max_batch_size) {
    lock.unlock();
    cv.notify_one();
  }

  return future;
}

void EvaluationQueue::run() noexcept {
  std::unique_lock lock { mutex };
  cv.wait_for(lock, std::chrono::milliseconds(25), [&] {
    return queue.size() >= max_batch_size;
  });
  size_t batch_count = (queue.size() + max_batch_size - 1) / max_batch_size;
  lock.unlock();

  for (int i = 0; i < batch_count; ++i) {
    lock.lock();
    size_t batch_size = std::min(queue.size(), max_batch_size);
    std::vector<std::shared_ptr<Container>> containers(batch_size);
    std::vector<std::promise<Result_t>> promises(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      containers[i] = queue.front().first;
      promises[i] = std::move(queue.front().second);
      queue.pop();
    }
    lock.unlock();

    auto [priors, values] = evaluate(containers);
    auto priors_ptr = static_cast<float*>(priors.request().ptr);
    auto values_ptr = static_cast<float*>(values.request().ptr);
    for (size_t i = 0; i < batch_size; ++i) {
      Result_t result;
      result.first.resize(Container::action_count);
      std::memcpy(result.first.data(), priors_ptr, sizeof(float) * Container::action_count);
      priors_ptr += Container::action_count;

      result.second = values_ptr[0];
      promises[i].set_value(std::move(result));
    }
  }
}