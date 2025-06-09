#pragma once
#include <array>

template <typename T, size_t C, size_t N, size_t M>
class Array3D {
public:
  Array3D(T val = {}) {
    m_data.fill(val);
  }

  template <typename Self>
  auto& operator[](this Self&& self, size_t i, size_t j, size_t k) noexcept {
    return self.m_data[i * N * M + j * M + k];
  }

  constexpr size_t size() const noexcept {
    return m_data.size();
  }

  template <size_t D>
  constexpr size_t size() const noexcept {
    if constexpr (D == 0) {
      return C;
    } else if constexpr (D == 1) {
      return N;
    } else if constexpr (D == 2) {
      return M;
    } else {
      std::unreachable();
    }
  }

  template <typename Self>
  auto data(this Self&& self) noexcept {
    return self.m_data.data();
  }

private:
  std::array<T, C * N * M> m_data;
};