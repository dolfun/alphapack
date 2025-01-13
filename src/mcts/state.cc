#include "state.h"
#include <map>
#include <cassert>
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
  auto mask = get_valid_state_mask(m_items.front());
  for (int x = 0; x < mask.rows(); ++x) {
    for (int y = 0; y < mask.cols(); ++y) {
      if (mask(x, y) >= 0) {
        actions.push_back(x * mask.cols() + y);
      }
    }
  }
  return actions;
}

void State::transition(int action_idx) {
  assert(action_idx >= 0 && action_idx < State::action_count);
  auto shape = m_items.front().shape;
  m_items.front().placed = true;
  std::rotate(m_items.begin(), m_items.begin() + 1, m_items.end());
  int x0 = action_idx / State::bin_length, y0 = action_idx % State::bin_length;
  for (int x = x0; x < x0 + shape.x; ++x) {
    for (int y = y0; y < y0 + shape.y; ++y) {
      m_height_map(x, y) += shape.z;
      assert(m_height_map(x, y) <= m_height);
    }
  }
}

float State::reward() const noexcept {
  float total_volume = 0.0f;
  for (auto pkg : m_items) {
    if (pkg.placed) total_volume += pkg.shape.x * pkg.shape.y * pkg.shape.z;
  }
  float packing_efficiency = total_volume / (State::bin_length * State::bin_length * State::bin_height);
  return packing_efficiency;
}

auto State::serialize() const noexcept -> std::string {
  std::pair<const void*, size_t> infos[] = {
    { m_items.data(), sizeof(Item) * State::item_count },
    { m_height_map.data(), sizeof(int) * State::bin_length * State::bin_length }
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

  std::pair<void*, size_t> infos[] = {
    { items.data(), sizeof(Item) * item_count },
    { height_map.data(), sizeof(int) * State::bin_length * State::bin_length }
  };

  const char* src = &bytes[0];
  for (auto [dest, size] : infos) {
    std::memcpy(dest, src, size);
    src += size;
  }

  return State(std::move(items), std::move(height_map));
}

auto State::get_valid_state_mask(const Item& pkg) const noexcept -> Array2D<int> {
  Array2D<int> mask { State::bin_length, State::bin_length, -1 };
  auto max_height_freq = get_max_freq_in_window(m_height_map, pkg.shape);
  for (std::size_t x = 0; x <= mask.rows() - pkg.shape.x; ++x) {
    for (std::size_t y = 0; y <= mask.cols() - pkg.shape.y; ++y) {
      auto [max_height, freq] = max_height_freq(x, y);
      if (max_height + pkg.shape.z > State::bin_height) continue;

      float min_base_contact_ratio = 0.00f;
      float base_contact_ratio = static_cast<float>(freq) / (pkg.shape.x * pkg.shape.y);
      if (base_contact_ratio >= min_base_contact_ratio) {
        mask(x, y) = max_height;
      }
    }
  }
  return mask;
}