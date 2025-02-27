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
  using Input = std::shared_ptr<State::InferInput>;
  using Result = std::unique_ptr<State::InferResult>;
  using InferFunc = std::function<
    std::pair<pybind11::array_t<float>, pybind11::array_t<float>>(const std::vector<Input>&)
  >;

  InferenceQueue(size_t _max_batch_size, InferFunc _infer_func)
    : max_batch_size { _max_batch_size }, infer_func { _infer_func } {}

  auto infer(Input) noexcept -> std::future<Result>;
  void run() noexcept;

private:
  using Element_t = std::pair<Input, std::promise<Result>>;

  size_t max_batch_size;
  InferFunc infer_func;

  std::mutex mutex;
  std::condition_variable cv;
  std::queue<Element_t> queue;
};