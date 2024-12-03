#pragma once
#include "package.h"
#include "array2d.h"
#include "package_generator.h"

class Container {
public:
  Container(int height, const std::vector<Package>&, PackageGenerateInfo);
  auto height() const noexcept -> int;
  auto packages() const noexcept -> const std::vector<Package>&;
  auto height_map() const noexcept -> const Array2D<int>&;

  auto possible_actions() const -> std::vector<int>;
  void transition(int);
  float reward() const noexcept;
  auto flatten() const noexcept -> std::vector<float>;

  static constexpr size_t action_count = 32;
  static constexpr int length = 16;
  static constexpr int width = 16;

private:
  auto get_valid_state_mask(const Package&, int) const noexcept -> Array2D<int>;
  void place_package(Package&, glm::ivec3, int) noexcept;

  int m_height;
  std::vector<Package> m_packages;
  PackageGenerateInfo m_generate_info;
  Array2D<int> m_height_map;
  mutable std::vector<std::pair<glm::ivec3, int>> first_fit_info;
};