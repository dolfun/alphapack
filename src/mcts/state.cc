#include "state.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

template <typename T, size_t N>
class SingleUseMaxQueue {
public:
  constexpr SingleUseMaxQueue() : it_front { m_data.begin() }, it_back { m_data.begin() } {
  }

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
  std::array<T, N> m_data;
  std::array<T, N>::iterator it_front, it_back;
};

template <typename T, size_t N, size_t M>
auto get_max_in_window(const Array2D<T, N, M>& arr, int length, int width) -> State::Array2D<T> {
  State::Array2D<T> res {};
  for (size_t x = 0; x < N; ++x) {
    SingleUseMaxQueue<T, M> max_queue {};
    for (int y = 0; y < width; ++y) {
      max_queue.insert(arr[x, y]);
    }

    for (size_t y = 0; y < M - width; ++y) {
      res[x, y] = max_queue.max();
      max_queue.remove(arr[x, y]);
      max_queue.insert(arr[x, y + width]);
    }

    res[x, M - width] = max_queue.max();
  }

  for (size_t y = 0; y <= M - width; ++y) {
    SingleUseMaxQueue<T, N> max_queue {};
    for (int x = 0; x < length; ++x) {
      max_queue.insert(res[x, y]);
    }

    for (size_t x = 0; x < N - length; ++x) {
      T item_to_remove = res[x, y];
      res[x, y] = max_queue.max();
      max_queue.remove(item_to_remove);
      max_queue.insert(res[x + length, y]);
    }

    res[N - length, y] = max_queue.max();
  }

  return res;
}

State::State(const std::vector<Item>& items)
    : m_feasibility_info { create_feasibility_info(items.front()) } {
  if (items.size() != item_count) {
    throw std::runtime_error("Invalid number of items received!");
  }

  std::copy(items.begin(), items.end(), m_items.begin());
}

auto State::items() const noexcept -> const std::array<Item, item_count>& {
  return m_items;
}

auto State::height_map() const noexcept -> const Array2D<int8_t>& {
  return m_height_map;
}

auto State::feasibility_mask() const noexcept -> Array2D<int8_t> {
  State::Array2D<int8_t> mask {};
  for (size_t x = 0; x < mask.size<0>(); ++x) {
    for (size_t y = 0; y < mask.size<1>(); ++y) {
      mask[x, y] = (m_feasibility_info[x, y] >= 0);
    }
  }
  return mask;
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
  for (int x = 0; x < m_feasibility_info.size<0>(); ++x) {
    for (int y = 0; y < m_feasibility_info.size<1>(); ++y) {
      if (m_feasibility_info[x, y] >= 0) {
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
      m_height_map[x, y] = m_feasibility_info[x0, y0] + current_item.shape.z;
    }
  }

  m_feasibility_info = create_feasibility_info(m_items.front());

  int used_items_count = 0;
  for (auto item : m_items) {
    used_items_count += static_cast<int>(item.placed);
  }

  int reward_scaling = item_count * (item_count + 1) / 2;
  float reward = static_cast<float>(used_items_count) / reward_scaling;
  return reward;
}

auto State::serialize(const State& state) -> std::string {
  std::pair<const void*, size_t> infos[3] = {
    { state.m_items.data(), sizeof(Item) * item_count },
    { state.m_height_map.data(), sizeof(int8_t) * state.m_height_map.size() },
    { state.m_feasibility_info.data(), sizeof(int8_t) * state.m_feasibility_info.size() }
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
  std::array<Item, item_count> items {};
  Array2D<int8_t> height_map {}, feasibility_info {};

  std::pair<void*, size_t> infos[3] = { { items.data(), sizeof(Item) * item_count },
                                        { height_map.data(), sizeof(int8_t) * height_map.size() },
                                        { feasibility_info.data(),
                                          sizeof(int8_t) * feasibility_info.size() } };

  const char* src = &bytes[0];
  for (auto [dest, size] : infos) {
    std::memcpy(dest, src, size);
    src += size;
  }

  return State(items, height_map, feasibility_info);
}

State::State(
  const std::array<Item, item_count>& items,
  const Array2D<int8_t>& height_map,
  const Array2D<int8_t>& feasibility_info
)
    : m_items { items }, m_height_map { height_map }, m_feasibility_info { feasibility_info } {
}

auto State::get_additional_data(bool swap) const noexcept
  -> std::array<float, additional_input_count> {
  std::array<float, additional_input_count> data;
  auto it = data.begin();
  for (auto item : m_items) {
    float x = static_cast<float>(item.shape.x) / bin_length;
    float y = static_cast<float>(item.shape.y) / bin_length;
    float z = static_cast<float>(item.shape.z) / bin_height;

    if (swap) std::swap(x, y);

    it[0] = x;
    it[1] = y;
    it[2] = z;
    it[3] = (item.placed ? 1.0f : 0.0f);

    it += 4;
  }

  return data;
}

auto State::inference_input(int k) const noexcept -> std::shared_ptr<InferInput> {
  auto inference_input = std::make_shared<InferInput>();
  inference_input->additional_data = get_additional_data(k & 1);

  auto current_item = m_items.front();
  int l = current_item.shape.x, w = current_item.shape.y;
  constexpr int L = State::bin_length;
  for (int x = 0; x < L; ++x) {
    for (int y = 0; y < L; ++y) {
      auto [x1, y1] = symmetric_transforms[k](x, y, 1, 1, L);
      inference_input->image_data[0, x1, y1] = static_cast<float>(m_height_map[x, y]) / bin_height;

      if (!current_item.placed && x <= L - l && y <= L - w) {
        auto [x2, y2] = symmetric_transforms[k](x, y, l, w, L);
        inference_input->image_data[1, x2, y2] = static_cast<float>(m_feasibility_info[x, y] >= 0);
      }
    }
  }

  return inference_input;
};

auto State::invert_symmetric_transform(
  const std::array<float, State::action_count>& priors,
  int k
) const noexcept -> std::array<float, State::action_count> {
  Item current_item = m_items.front();
  if (current_item.placed) return {};  // !?

  constexpr int L = State::bin_length;
  int l = current_item.shape.x, w = current_item.shape.y;

  std::array<float, State::action_count> inverted_priors {};
  for (int x = 0; x <= L - l; ++x) {
    for (int y = 0; y <= L - w; ++y) {
      auto [x1, y1] = symmetric_transforms[k](x, y, l, w, L);
      int idx = x * L + y;
      int idx1 = x1 * L + y1;
      inverted_priors[idx] = priors[idx1];
    }
  }

  return inverted_priors;
}

auto State::create_feasibility_info(const Item& item) const noexcept -> Array2D<int8_t> {
  Array2D<int8_t> info { -1 };
  if (item.placed) return info;

  auto max_height_arr = get_max_in_window(m_height_map, item.shape.x, item.shape.y);
  for (size_t x = 0; x <= info.size<0>() - item.shape.x; ++x) {
    for (size_t y = 0; y <= info.size<1>() - item.shape.y; ++y) {
      int max_height = max_height_arr[x, y];
      if (max_height + item.shape.z <= bin_height) {
        info[x, y] = max_height;
      }
    }
  }

  return info;
}