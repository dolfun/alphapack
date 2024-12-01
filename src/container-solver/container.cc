#include "container.h"
#include <map>
#include <cassert>

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

auto ContainerState::get_packages() const noexcept -> const std::vector<Package>& {
  return packages;
}

auto ContainerState::possible_actions() const -> std::vector<int> {
  std::vector<int> actions;

  for (int i = 0; i < std::ssize(packages); ++i) {
    const auto& pkg = packages[i];
    if (pkg.is_placed) continue;

    for (int orientation = 5; orientation >= 0; --orientation) {
      bool found = false;
      auto mask = get_valid_state_mask(pkg, orientation);
      for (int x = 0; x < mask.nr_rows(); ++x) {
        for (int y = 0; y < mask.nr_cols(); ++y) {
          if (mask[x, y] >= 0) {
            glm::ivec3 pos = { x, y, mask[x, y] };
            first_fit_info[i] = { pos, orientation };
            actions.push_back(i);
            found = true;
            break;
          }
        }
        if (found) break;
      }
      if (found) break;
    }
  }

  return actions;
}

void ContainerState::transition(int action_idx) {
  auto [pos, orientation] = first_fit_info[action_idx];
  place_package(packages[action_idx], pos, orientation);
}

ContainerState::ContainerState(ContainerInfo info, const std::vector<Package>& _packages)
  : container_info { info }, packages { _packages },
    height_map { ContainerInfo::length, ContainerInfo::width },
    first_fit_info(ContainerInfo::max_pkg_cnt) {}

auto ContainerState::get_valid_state_mask(const Package& pkg, int orientation) const noexcept -> Array2D<int> {
  Array2D<int> mask { ContainerInfo::length, ContainerInfo::width, -1 };
  auto shape = Package::get_shape_along_axes(pkg.shape, orientation);
  auto max_height_freq = get_max_freq_in_window(height_map, shape);
  for (std::size_t x = 0; x <= mask.nr_rows() - shape.x; ++x) {
    for (std::size_t y = 0; y <= mask.nr_cols() - shape.y; ++y) {
      auto [max_height, freq] = max_height_freq[x, y];
      if (max_height + shape.z > container_info.height) continue;

      float MIN_BASE_CONTACT_RATIO = 0.75f;
      float base_contact_ratio = static_cast<float>(freq) / (shape.x * shape.y);
      if (base_contact_ratio >= MIN_BASE_CONTACT_RATIO) {
        mask[x, y] = max_height;
      }
    }
  }
  return mask;
}

void ContainerState::place_package(Package& pkg, glm::ivec3 pos, int orientation) noexcept {
  assert(!pkg.is_placed);
  pkg.is_placed = true;
  pkg.shape = Package::get_shape_along_axes(pkg.shape, orientation);
  pkg.pos = pos;

  auto pos2 = pkg.pos + pkg.shape;
  for (int x = pkg.pos.x; x < pos2.x; ++x) {
    for (int y = pkg.pos.y; y < pos2.y; ++y) {
      height_map[x, y] = pos2.z;
    }
  }
}