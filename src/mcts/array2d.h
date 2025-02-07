#pragma once
#include <vector>
#include <utility>

template <typename T>
class Array2D {
public:
  Array2D(size_t rows, size_t cols, T val = {})
    : m_rows { rows }, m_cols { cols }, m_data(m_rows * m_cols, val) {}

  const T& operator() (size_t i, size_t j) const noexcept {
    return m_data[i * m_cols + j];
  }

  T& operator() (size_t i, size_t j) noexcept {
    return const_cast<T&>(std::as_const(*this).operator()(i, j));
  }

  size_t rows() const noexcept {
    return m_rows;
  }

  size_t cols() const noexcept {
    return m_cols;
  }

  const T* data() const noexcept {
    return m_data.data();
  }

  T* data() noexcept {
    return const_cast<T*>(std::as_const(*this).data());
  }

private:
  size_t m_rows, m_cols;
  std::vector<T> m_data;
};