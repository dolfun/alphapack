#include "state.h"
#include <map>
#include <cstring>
#include <algorithm>

// Optimize further using max queue
template <typename T>
auto get_max_freq_in_window(const Array2D<T>& arr, Vec3i shape) -> Array2D<std::pair<T, int>> {
  std::size_t n_arr = arr.rows(), m_arr = arr.cols();
  std::size_t n_shape = shape.x, m_shape = shape.y;

  Array2D<std::pair<T, int>> res { n_arr, m_arr };
  for (std::size_t x = 0; x < n_arr; ++x) {
    std::map<int, int, std::greater<>> freq;
    for (std::size_t y = 0; y < m_shape; ++y) {
      ++freq[arr(x, y)];
    }

    for (std::size_t y = 0; y < m_arr - m_shape; ++y) {
      res(x, y) = *freq.begin();

      int val = arr(x, y);
      if (--freq[val] == 0) {
        freq.erase(val);
      }

      ++freq[arr(x, y + m_shape)];
    }
    res(x, m_arr - m_shape) = *freq.begin();
  }

  for (std::size_t y = 0; y <= m_arr - m_shape; ++y) {
    std::map<int, int, std::greater<>> freq;
    for (std::size_t x = 0; x < n_shape; ++x) {
      freq[res(x, y).first] += res(x, y).second;
    }

    for (std::size_t x = 0; x < n_arr - n_shape; ++x) {
      auto [val, cnt] = res(x, y);
      res(x, y) = *freq.begin();

      freq[val] -= cnt;
      if (freq[val] == 0) {
        freq.erase(val);
      }

      freq[res(x + n_shape, y).first] += res(x + n_shape, y).second;
    }
    res(n_arr - n_shape, y) = *freq.begin();
  }

  return res;
}

auto State::items() const noexcept -> const std::vector<Item>& {
  return m_items;
}

auto State::height_map() const noexcept -> const Array2D<int>& {
  return m_height_map;
}

auto State::feasibility_mask() const noexcept -> const Array2D<char> {
  Array2D<char> mask { m_feasibility_info.rows(), m_feasibility_info.cols() };
  for (size_t i = 0; i < mask.rows(); ++i) {
    for (size_t j = 0; j < mask.cols(); ++j) {
      mask(i, j) = (m_feasibility_info(i, j) >= 0);
    }
  }
  return mask;
}

auto State::normalized_items() const noexcept -> const std::vector<float> {
  std::vector<float> data(item_count * values_per_item);
  auto it = data.begin();
  for (const auto& item : m_items) {
    if (!item.placed) {
      float x = static_cast<float>(item.shape.x) / State::bin_length;
      float y = static_cast<float>(item.shape.y) / State::bin_length;
      float z = static_cast<float>(item.shape.z) / State::bin_height;
      it[0] = x;
      it[1] = y;
      it[2] = z;
      it[3] = x * y * z;
    }
    it += 4;
  }
  return data;
}

auto State::possible_actions() const -> std::vector<int> {
  std::vector<int> actions;
  for (int x = 0; x < m_feasibility_info.rows(); ++x) {
    for (int y = 0; y < m_feasibility_info.cols(); ++y) {
      if (m_feasibility_info(x, y) >= 0) {
        actions.push_back(x * m_feasibility_info.cols() + y);
      }
    }
  }
  return actions;
}

void State::transition(int action_idx) {
  Item current_item = m_items.front();
  current_item.placed = true;
  std::rotate(m_items.begin(), m_items.begin() + 1, m_items.end());

  int x0 = action_idx / State::bin_length, y0 = action_idx % State::bin_length;
  for (int x = x0; x < x0 + current_item.shape.x; ++x) {
    for (int y = y0; y < y0 + current_item.shape.y; ++y) {
      m_height_map(x, y) = m_feasibility_info(x0, y0) + current_item.shape.z;
    }
  }

  m_feasibility_info = create_feasibility_info(m_items.front());
}

float State::reward() const noexcept {
  float total_volume = 0.0f;
  for (auto item : m_items) {
    if (item.placed) total_volume += item.shape.x * item.shape.y * item.shape.z;
  }
  float packing_efficiency = total_volume / (State::bin_length * State::bin_length * State::bin_height);
  return packing_efficiency;
}

auto State::serialize(const State& state) -> std::string {
  std::pair<const void*, size_t> infos[3] = {
    { state.m_items.data(), sizeof(Item) * State::item_count },
    { state.m_height_map.data(), sizeof(int) * State::bin_length * State::bin_length },
    { state.m_feasibility_info.data(), sizeof(int) * State::bin_length * State::bin_length }
  };

  size_t total_size = 0;
  for (auto [_, size] : infos) {
    total_size += size;
  }

  std::string bytes(total_size, ' ');
  char* dest = &bytes[0];
  for (auto [src, size] : infos) {
    std::memcpy(dest, src, size);
    dest += size;
  }
  return bytes;
}

State State::unserialize(const std::string& bytes) {
  std::vector<Item> items(item_count);
  Array2D<int> height_map(State::bin_length, State::bin_length);
  Array2D<int> feasibility_info(State::bin_length, State::bin_length);

  std::pair<void*, size_t> infos[3] = {
    { items.data(), sizeof(Item) * item_count },
    { height_map.data(), sizeof(int) * State::bin_length * State::bin_length },
    { feasibility_info.data(), sizeof(int) * State::bin_length * State::bin_length }
  };

  const char* src = &bytes[0];
  for (auto [dest, size] : infos) {
    std::memcpy(dest, src, size);
    src += size;
  }

  return State(std::move(items), std::move(height_map), std::move(feasibility_info));
}

auto State::create_feasibility_info(const Item& item) const noexcept -> Array2D<int> {
  Array2D<int> info { State::bin_length, State::bin_length, -1 };
  if (item.placed) return info;

  auto max_height_freq = get_max_freq_in_window(m_height_map, item.shape);
  for (std::size_t x = 0; x <= info.rows() - item.shape.x; ++x) {
    for (std::size_t y = 0; y <= info.cols() - item.shape.y; ++y) {
      auto [max_height, _] = max_height_freq(x, y);
      if (max_height + item.shape.z > State::bin_height) continue;
      info(x, y) = max_height;
    }
  }
  return info;
}