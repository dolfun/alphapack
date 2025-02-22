#pragma once
#include <array>

template <typename T, size_t N, size_t M>
class Array2D {
public:
  Array2D(T val = {}) { m_data.fill(val); }

  template <typename Self>
  auto& operator[] (this Self&& self, size_t i, size_t j) noexcept {
    return self.m_data[i * M + j];
  }

  constexpr size_t rows() const noexcept {
    return N;
  }

  constexpr size_t cols() const noexcept {
    return M;
  }

  template <typename Self>
  auto data(this Self&& self) noexcept {
    return self.m_data.data();
  }

private:
  std::array<T, N * M> m_data;
};