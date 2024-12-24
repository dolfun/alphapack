#pragma once
#include <string>
#include "package.h"
#include "array2d.h"

class Container {
public:
  Container(int height, const std::vector<Package>& packages)
    : m_height { height }, m_packages { packages },
      m_height_map { Container::length, Container::length } {}

  auto height() const noexcept -> int;
  auto packages() const noexcept -> const std::vector<Package>&;
  auto height_map() const noexcept -> const Array2D<int>&;

  auto normalized_packages() const noexcept -> const std::vector<float>;

  auto possible_actions() const -> std::vector<int>;
  void transition(int);
  float reward() const noexcept;

  auto serialize() const noexcept -> std::string;
  static Container unserialize(const std::string&);

  static constexpr int length = 16;
  static constexpr size_t action_count = length * length;
  static constexpr size_t package_count = 32;
  static constexpr size_t values_per_package = 4;

private:
  Container(int height, std::vector<Package>&& packages, Array2D<int>&& height_map)
    : m_height { height }, m_packages { std::move(packages) }, m_height_map { std::move(height_map) } {}
  
  auto get_valid_state_mask(const Package&) const noexcept -> Array2D<int>;

  int m_height;
  std::vector<Package> m_packages;
  Array2D<int> m_height_map;
};