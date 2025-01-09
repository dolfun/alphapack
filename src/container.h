#pragma once
#include <string>
#include "package.h"
#include "array2d.h"

class Container {
public:
  Container(const std::vector<Package>& packages)
    : m_packages { packages },
      m_height_map { Container::length, Container::length } {}

  auto packages() const noexcept -> const std::vector<Package>&;
  auto height_map() const noexcept -> const Array2D<int>&;

  auto normalized_packages() const noexcept -> const std::vector<float>;

  auto possible_actions() const -> std::vector<int>;
  void transition(int);
  float reward() const noexcept;

  auto serialize() const noexcept -> std::string;
  static Container unserialize(const std::string&);

  static constexpr int length = 10;
  static constexpr int height = 10;
  static constexpr size_t action_count = length * length;
  static constexpr size_t package_count = 32;
  static constexpr size_t values_per_package = 4;

private:
  Container(std::vector<Package>&& packages, Array2D<int>&& height_map)
    : m_packages { std::move(packages) }, m_height_map { std::move(height_map) } {}
  
  auto get_valid_state_mask(const Package&) const noexcept -> Array2D<int>;

  std::vector<Package> m_packages;
  Array2D<int> m_height_map;
};