#pragma once
#include "package.h"
#include "array2d.h"

struct ContainerInfo {
  int height;
  int weight_limit;

  static constexpr int length = 16;
  static constexpr int width = 16;
  static constexpr int max_pkg_cnt = 32;
};

class ContainerState {
public:
  ContainerState(ContainerInfo, const std::vector<Package>&);
  auto get_packages() const noexcept -> const std::vector<Package>&;

  auto possible_actions() const -> std::vector<int>;
  void transition(int);

private:
  auto get_valid_state_mask(const Package&, int) const noexcept -> Array2D<int>;
  void place_package(Package&, glm::ivec3, int) noexcept;

  ContainerInfo container_info;
  std::vector<Package> packages;
  Array2D<int> height_map;
  mutable std::vector<std::pair<glm::ivec3, int>> first_fit_info;
};