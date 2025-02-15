#include "state.h"
#include <cstring>
#include <algorithm>

template <typename T>
class SingleUseMaxQueue {
public:
  constexpr SingleUseMaxQueue(size_t size)
    : m_data(size), it_front { m_data.begin() }, it_back { m_data.begin() } {}

  constexpr T max() const noexcept {
    return *it_front;
  }

  constexpr void insert(T item) noexcept {
    while (it_front < it_back && *std::prev(it_back) < item) {
      --it_back;
    }
    *(it_back++) = item;
  }

  constexpr void remove(T item) noexcept {
    if (it_front < it_back && *it_front == item) {
      ++it_front;
    }
  }

private:
  std::vector<T> m_data;
  std::vector<T>::iterator it_front, it_back;
};

template <typename T>
auto get_max_in_window(const Array2D<T>& arr, int length, int width) -> Array2D<T> {
  size_t n_arr = arr.rows(), m_arr = arr.cols();

  Array2D<T> res { n_arr, m_arr };
  for (size_t x = 0; x < n_arr; ++x) {
    SingleUseMaxQueue<T> max_queue { m_arr };
    for (int y = 0; y < width; ++y) {
      max_queue.insert(arr(x, y));
    }

    for (size_t y = 0; y < m_arr - width; ++y) {
      res(x, y) = max_queue.max();
      max_queue.remove(arr(x, y));
      max_queue.insert(arr(x, y + width));
    }

    res(x, m_arr - width) = max_queue.max();
  }

  for (size_t y = 0; y <= m_arr - width; ++y) {
    SingleUseMaxQueue<T> max_queue { n_arr };
    for (int x = 0; x < length; ++x) {
      max_queue.insert(res(x, y));
    }

    for (size_t x = 0; x < n_arr - length; ++x) {
      T item_to_remove = res(x, y);
      res(x, y) = max_queue.max();
      max_queue.remove(item_to_remove);
      max_queue.insert(res(x + length, y));
    }

    res(n_arr - length, y) = max_queue.max();
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
  std::vector<float> data(item_count * values_per_item);
  auto it = data.begin();
  for (const auto& item : m_items) {
    if (!item.placed) {
      float x = static_cast<float>(item.shape.x) / bin_length;
      float y = static_cast<float>(item.shape.y) / bin_length;
      float z = static_cast<float>(item.shape.z) / bin_height;

      it[0] = x;
      it[1] = y;
      it[2] = z;
    }

    it += values_per_item;
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
        actions.push_back(x * State::bin_length + y);
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

  int used_items_count = 0;
  for (auto item : m_items) {
    used_items_count += static_cast<int>(item.placed);
  }

  int reward_scaling = item_count * (item_count) / 2;
  float reward = static_cast<float>(used_items_count) / reward_scaling;
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

  auto max_height_arr = get_max_in_window(m_height_map, item.shape.x, item.shape.y);
  for (size_t x = 0; x <= info.rows() - item.shape.x; ++x) {
    for (size_t y = 0; y <= info.cols() - item.shape.y; ++y) {
      int max_height = max_height_arr(x, y);
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
  Item current_item = items.front();
  int l = current_item.shape.x, w = current_item.shape.y;

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
  if (!current_item.placed) {
    for (int x = 0; x <= L - l; ++x) {
      for (int y = 0; y <= L - w; ++y) {
        auto [x1, y1] = symmetry_transforms[k](x, y, l, w, L);
        feasibility_info(x1, y1) = state.m_feasibility_info(x, y);
      }
    }
  }

  return State(std::move(items), std::move(height_map), std::move(feasibility_info));
}

auto get_inverse_priors_symmetry(const State& state, const std::vector<float>& transformed_priors, int k) noexcept
    -> std::vector<float> {

  Item current_item = state.items().front();
  if (current_item.placed) return {};

  constexpr int L = State::bin_length;
  int l = current_item.shape.x, w = current_item.shape.y;

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