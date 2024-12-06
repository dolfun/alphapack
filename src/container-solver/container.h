#pragma once
#include <string>
#include "package.h"
#include "array2d.h"

class Container {
public:
  Container(int, const std::vector<Package>&);

  auto height() const noexcept -> int;
  auto packages() const noexcept -> const std::vector<Package>&;
  auto height_map() const noexcept -> const Array2D<int>&;

  auto possible_actions() const -> std::vector<int>;
  void transition(int);
  float reward() const noexcept;

  auto serialize() const noexcept -> std::string;
  static Container unserialize(const std::string&);

  static constexpr size_t action_count = 32;
  static constexpr int length = 16;
  static constexpr int width = 16;

private:
  Container(int, std::vector<Package>&&, Array2D<int>&&);

  auto get_valid_state_mask(const Package&, int) const noexcept -> Array2D<int>;
  void place_package(Package&, glm::ivec3, int) noexcept;

  int m_height;
  std::vector<Package> m_packages;
  Array2D<int> m_height_map;
};