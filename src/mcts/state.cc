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

auto State::height_map() const noexcept -> const Array2D<int8_t>& {
  return m_height_map;
}

auto State::feasibility_mask() const noexcept -> Array2D<int8_t> {
  Array2D<int8_t> mask { m_feasibility_info.rows(), m_feasibility_info.cols() };
  for (size_t x = 0; x < mask.rows(); ++x) {
    for (size_t y = 0; y < mask.cols(); ++y) {
      mask(x, y) = (m_feasibility_info(x, y) >= 0);
    }
  }
  return mask;
}

auto State::normalized_items() const noexcept -> std::vector<float> {
  auto normalize_dim = [] (int val) {
    return (static_cast<float>(val) - min_item_dim) / (max_item_dim - min_item_dim);
  };

  auto normalize_vol = [] (int val) {
    constexpr float min_item_vol = (min_item_dim * min_item_dim * min_item_dim);
    constexpr float max_item_vol = (max_item_dim * max_item_dim * max_item_dim);
    return (static_cast<float>(val) - min_item_vol) / (max_item_vol - min_item_vol);
  };

  std::vector<float> data(item_count * values_per_item);
  auto it = data.begin();
  for (const auto& item : m_items) {
    if (!item.placed) {
      it[0] = normalize_dim(item.shape.x);
      it[1] = normalize_dim(item.shape.y);
      it[2] = normalize_dim(item.shape.z);
      it[3] = normalize_vol(item.shape.x * item.shape.y * item.shape.z);
    }
    it += 4;
  }
  return data;
}

float State::packing_efficiency() const noexcept {
  int total_volume = 0;
  for (const auto& item : m_items) {
    if (item.placed) {
      total_volume += item.shape.x * item.shape.y * item.shape.z;
    }
  }

  constexpr int bin_volume = (bin_length * bin_length * bin_height);
  return static_cast<float>(total_volume) / bin_volume;
}

auto State::possible_actions() const -> std::vector<int> {
  if (m_items.front().placed) return {};

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

float State::transition(int action_idx) {
  std::rotate(m_items.begin(), m_items.begin() + 1, m_items.end());
  Item& current_item = m_items.back();
  current_item.placed = true;

  int x0 = action_idx / bin_length, y0 = action_idx % bin_length;
  for (int x = x0; x < x0 + current_item.shape.x; ++x) {
    for (int y = y0; y < y0 + current_item.shape.y; ++y) {
      m_height_map(x, y) = m_feasibility_info(x0, y0) + current_item.shape.z;
    }
  }

  m_feasibility_info = create_feasibility_info(m_items.front());

  float reward = current_item.shape.x * current_item.shape.y * current_item.shape.z;
  reward /= (bin_length * bin_length * bin_height);
  return reward;
}

auto State::serialize(const State& state) -> std::string {
  std::pair<const void*, size_t> infos[3] = {
    { state.m_items.data(), sizeof(Item) * item_count },
    { state.m_height_map.data(), sizeof(int8_t) * bin_length * bin_length },
    { state.m_feasibility_info.data(), sizeof(int8_t) * bin_length * bin_length }
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
  Array2D<int8_t> height_map(bin_length, bin_length);
  Array2D<int8_t> feasibility_info(bin_length, bin_length);

  std::pair<void*, size_t> infos[3] = {
    { items.data(), sizeof(Item) * item_count },
    { height_map.data(), sizeof(int8_t) * bin_length * bin_length },
    { feasibility_info.data(), sizeof(int8_t) * bin_length * bin_length }
  };

  const char* src = &bytes[0];
  for (auto [dest, size] : infos) {
    std::memcpy(dest, src, size);
    src += size;
  }

  return State(std::move(items), std::move(height_map), std::move(feasibility_info));
}

auto State::create_feasibility_info(const Item& item) const noexcept -> Array2D<int8_t> {
  Array2D<int8_t> info { bin_length, bin_length, -1 };
  if (item.placed) return info;

  auto max_height_freq = get_max_freq_in_window(m_height_map, item.shape);
  for (std::size_t x = 0; x <= info.rows() - item.shape.x; ++x) {
    for (std::size_t y = 0; y <= info.cols() - item.shape.y; ++y) {
      auto [max_height, _] = max_height_freq(x, y);
      if (max_height + item.shape.z > bin_height) continue;
      info(x, y) = max_height;
    }
  }
  return info;
}

constexpr std::pair<int, int> (*symmetry_transforms[8])(int, int, int, int, int) = {
  [] (int x, int y, int l, int w, int L) { return std::make_pair(x        , y        ); },
  [] (int x, int y, int l, int w, int L) { return std::make_pair(L - y - w, x        ); },
  [] (int x, int y, int l, int w, int L) { return std::make_pair(L - x - l, L - y - w); },
  [] (int x, int y, int l, int w, int L) { return std::make_pair(y        , L - x - l); },
  [] (int x, int y, int l, int w, int L) { return std::make_pair(L - x - l, y        ); },
  [] (int x, int y, int l, int w, int L) { return std::make_pair(L - y - w, L - x - l); },
  [] (int x, int y, int l, int w, int L) { return std::make_pair(x        , L - y - w); },
  [] (int x, int y, int l, int w, int L) { return std::make_pair(y        , x        ); },
};

auto get_state_symmetry(const State& state, int k) noexcept -> State {
  auto items = state.m_items;
  int l = items[0].shape.x, w = items[0].shape.y;

  if (k % 2 == 1) {
    for (auto& item : items) {
      std::swap(item.shape.x, item.shape.y);
    }
  }

  constexpr int L = State::bin_length;
  Array2D<int8_t> height_map { L, L };
  for (int x = 0; x < L; ++x) {
    for (int y = 0; y < L; ++y) {
      auto [x1, y1] = symmetry_transforms[k](x, y, 1, 1, L);
      height_map(x1, y1) = state.m_height_map(x, y);
    }
  }

  Array2D<int8_t> feasibility_info { L, L, -1 };
  for (int x = 0; x <= L - l; ++x) {
    for (int y = 0; y <= L - w; ++y) {
      auto [x1, y1] = symmetry_transforms[k](x, y, l, w, L);
      feasibility_info(x1, y1) = state.m_feasibility_info(x, y);
    }
  }

  return State(std::move(items), std::move(height_map), std::move(feasibility_info));
}

auto get_inverse_priors_symmetry(const State& state, const std::vector<float>& transformed_priors, int k) noexcept
    -> std::vector<float> {

  constexpr int L = State::bin_length;
  int l = state.items()[0].shape.x, w = state.items()[0].shape.y;

  std::vector<float> priors(transformed_priors.size());
  for (int x = 0; x <= L - l; ++x) {
    for (int y = 0; y <= L - w; ++y) {
      auto [x1, y1] = symmetry_transforms[k](x, y, l, w, L);
      int idx = x * L + y;
      int idx1 = x1 * L + y1;
      priors[idx] = transformed_priors[idx1];
    }
  }
  
  return priors;
}