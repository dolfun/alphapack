#pragma once
#include "state.h"
#include <pybind11/numpy.h>
#include <condition_variable>
#include <functional>
#include <memory>
#include <future>
#include <mutex>
#include <queue>

class InferenceQueue {
public:
  using Result = std::pair<std::vector<float>, float>;
  using InferFunc = std::function<
    std::pair<pybind11::array_t<float>, pybind11::array_t<float>>(const std::vector<std::shared_ptr<State>>&)
  >;

  InferenceQueue(size_t _max_batch_size, InferFunc _infer_func)
    : max_batch_size { _max_batch_size }, infer_func { _infer_func } {}

  auto infer(std::shared_ptr<State>) noexcept -> std::future<Result>;
  void run() noexcept;

private:
  using Element_t = std::pair<std::shared_ptr<State>, std::promise<Result>>;

  size_t max_batch_size;
  InferFunc infer_func;

  std::mutex mutex;
  std::condition_variable cv;
  std::queue<Element_t> queue;
};