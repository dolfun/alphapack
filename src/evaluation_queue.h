#pragma once
#include "container.h"
#include <pybind11/numpy.h>
#include <condition_variable>
#include <functional>
#include <memory>
#include <future>
#include <mutex>
#include <queue>

class EvaluationQueue {
public:
  using Result_t = std::pair<std::vector<float>, float>;
  using Evaluate_t = std::function<
    std::pair<pybind11::array_t<float>, pybind11::array_t<float>>(const std::vector<std::shared_ptr<Container>>&)
  >;

  EvaluationQueue(size_t _max_batch_size, Evaluate_t _evaluate)
    : max_batch_size { _max_batch_size }, evaluate { _evaluate } {}

  auto enqueue(std::shared_ptr<Container>) noexcept -> std::future<Result_t>;
  void run() noexcept;

private:
  using Element_t = std::pair<std::shared_ptr<Container>, std::promise<Result_t>>;

  size_t max_batch_size;
  Evaluate_t evaluate;

  std::mutex mutex;
  std::condition_variable cv;
  std::queue<Element_t> queue;
};