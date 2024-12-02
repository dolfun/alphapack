#include "container.h"
#include <map>
#include <cassert>

Container::Container(int height, const std::vector<Package>& _packages)
  : m_height { height }, m_packages { _packages },
    m_height_map { Container::length, Container::width },
    first_fit_info(action_count) {}

auto Container::height() const noexcept -> int {
  return m_height;
}

auto Container::packages() const noexcept -> const std::vector<Package>& {
  return m_packages;
}

auto Container::height_map() const noexcept -> const Array2D<int>& {
  return m_height_map;
}

auto Container::possible_actions() const -> std::vector<int> {
  std::vector<int> actions;

  for (int i = 0; i < std::ssize(m_packages); ++i) {
    const auto& pkg = m_packages[i];
    if (pkg.is_placed) continue;

    for (int orientation = 5; orientation >= 0; --orientation) {
      auto mask = get_valid_state_mask(pkg, orientation);
      for (int x = 0; x < mask.nr_rows(); ++x) {
        for (int y = 0; y < mask.nr_cols(); ++y) {
          if (mask[x, y] >= 0) {
            glm::ivec3 pos = { x, y, mask[x, y] };
            first_fit_info[i] = { pos, orientation };
            actions.push_back(i);
            goto exit;
          }
        }
      }
      exit:
    }
  }

  return actions;
}

void Container::transition(int action_idx) {
  auto [pos, orientation] = first_fit_info[action_idx];
  place_package(m_packages[action_idx], pos, orientation);
}

float Container::reward() const noexcept {
  float total_volume = 0.0f;
  for (auto pkg : m_packages) {
    if (pkg.is_placed) total_volume += pkg.shape.x * pkg.shape.y * pkg.shape.z;
  }
  float packing_efficiency = total_volume / (m_height * Container::length * Container::width);
  return packing_efficiency;
}

auto Container::flatten() const noexcept -> std::vector<float> {
  size_t height_map_size = m_height_map.data().size();
  size_t values_per_packages = 4;
  size_t packages_size = values_per_packages * m_packages.size();

  std::vector<float> data;
  data.reserve(height_map_size + packages_size);
  for (int x = 0; x < m_height_map.nr_rows(); ++x) {
    for (int y = 0; y < m_height_map.nr_cols(); ++y) {
      data.push_back(static_cast<float>(m_height_map[x, y]) / m_height);
    }
  }

  for (const auto& pkg : m_packages) {
    if (pkg.is_placed) {
      for (int i = 0; i < values_per_packages; ++i) {
        data.push_back(0.0f);
      }
    } else {
      data.push_back(pkg.shape.x);
      data.push_back(pkg.shape.y);
      data.push_back(pkg.shape.z);
      data.push_back(pkg.cost);
    }
  }

  return data;
}

template <typename T>
auto get_max_freq_in_window(const Array2D<T>& arr, glm::ivec3 shape) -> Array2D<std::pair<T, int>> {
  std::size_t n_arr = arr.nr_rows(), m_arr = arr.nr_cols();
  std::size_t n_shape = shape.x, m_shape = shape.y;

  Array2D<std::pair<T, int>> res { n_arr, m_arr };
  for (std::size_t x = 0; x < n_arr; ++x) {
    std::map<int, int, std::greater<>> freq;
    for (std::size_t y = 0; y < m_shape; ++y) {
      ++freq[arr[x, y]];
    }

    for (std::size_t y = 0; y < m_arr - m_shape; ++y) {
      res[x, y] = *freq.begin();

      int val = arr[x, y];
      if (--freq[val] == 0) {
        freq.erase(val);
      }

      ++freq[arr[x, y + m_shape]];
    }
    res[x, m_arr - m_shape] = *freq.begin();
  }

  for (std::size_t y = 0; y <= m_arr - m_shape; ++y) {
    std::map<int, int, std::greater<>> freq;
    for (std::size_t x = 0; x < n_shape; ++x) {
      freq[res[x, y].first] += res[x, y].second;
    }

    for (std::size_t x = 0; x < n_arr - n_shape; ++x) {
      auto [val, cnt] = res[x, y];
      res[x, y] = *freq.begin();

      freq[val] -= cnt;
      if (freq[val] == 0) {
        freq.erase(val);
      }

      freq[res[x + n_shape, y].first] += res[x + n_shape, y].second;
    }
    res[n_arr - n_shape, y] = *freq.begin();
  }

  return res;
}

auto Container::get_valid_state_mask(const Package& pkg, int orientation) const noexcept -> Array2D<int> {
  Array2D<int> mask { Container::length, Container::width, -1 };
  auto shape = Package::get_shape_along_axes(pkg.shape, orientation);
  auto max_height_freq = get_max_freq_in_window(m_height_map, shape);
  for (std::size_t x = 0; x <= mask.nr_rows() - shape.x; ++x) {
    for (std::size_t y = 0; y <= mask.nr_cols() - shape.y; ++y) {
      auto [max_height, freq] = max_height_freq[x, y];
      if (max_height + shape.z > m_height) continue;

      float min_base_contact_ratio = 0.75f;
      float base_contact_ratio = static_cast<float>(freq) / (shape.x * shape.y);
      if (base_contact_ratio >= min_base_contact_ratio) {
        mask[x, y] = max_height;
      }
    }
  }
  return mask;
}

void Container::place_package(Package& pkg, glm::ivec3 pos, int orientation) noexcept {
  assert(!pkg.is_placed);
  pkg.is_placed = true;
  pkg.shape = Package::get_shape_along_axes(pkg.shape, orientation);
  pkg.pos = pos;

  auto pos2 = pkg.pos + pkg.shape;
  for (int x = pkg.pos.x; x < pos2.x; ++x) {
    for (int y = pkg.pos.y; y < pos2.y; ++y) {
      m_height_map[x, y] = pos2.z;
    }
  }
}