#pragma once
#include <vector>

template <typename T>
class Array2D {
public:
  Array2D(std::size_t _n, std::size_t _m, T val = {})
    : n { _n }, m { _m }, v(n * m, val) {}

  template <typename Self>
  auto& operator[] (this Self&& self, std::size_t i, std::size_t j) noexcept {
    return self.v[i * self.m + j];
  }

  std::size_t nr_rows() const noexcept {
    return n;
  }

  std::size_t nr_cols() const noexcept {
    return m;
  }

  auto data() const noexcept -> const std::vector<T> {
    return v;
  }

private:
  std::size_t n, m;
  std::vector<T> v;
};