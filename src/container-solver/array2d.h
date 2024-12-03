#pragma once
#include <vector>

template <typename T>
class Array2D {
public:
  Array2D(std::size_t rows, std::size_t cols, T val = {})
    : m_rows { rows }, m_cols { cols }, m_data(m_rows * m_cols, val) {}

  template <typename Self>
  auto& operator[] (this Self&& self, std::size_t i, std::size_t j) noexcept {
    return self.m_data[i * self.m_cols + j];
  }

  std::size_t rows() const noexcept {
    return m_rows;
  }

  std::size_t cols() const noexcept {
    return m_cols;
  }

  template <typename Self>
  auto data(this Self&& self) noexcept {
    return self.m_data.data();
  }

private:
  std::size_t m_rows, m_cols;
  std::vector<T> m_data;
};