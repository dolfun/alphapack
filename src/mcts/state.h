#pragma once
#include <string>
#include "item.h"
#include "array2d.h"

class State {
public:
  State(const std::vector<Item>& items)
    : m_items { items },
      m_height_map { State::bin_length, State::bin_length },
      m_feasibility_info { create_feasibility_info(m_items.front()) } {}

  auto items() const noexcept -> const std::vector<Item>&;
  auto height_map() const noexcept -> const Array2D<int>&;
  auto feasibility_mask() const noexcept -> const Array2D<char>;

  auto normalized_items() const noexcept -> const std::vector<float>;

  auto possible_actions() const -> std::vector<int>;
  void transition(int);
  float reward() const noexcept;

  static std::string serialize(const State&);
  static State unserialize(const std::string&);

  static constexpr int bin_length = 10;
  static constexpr int bin_height = 10;
  static constexpr size_t action_count = bin_length * bin_length;
  static constexpr size_t item_count = 32;
  static constexpr size_t values_per_item = 4;

private:
  State(std::vector<Item>&& items, Array2D<int>&& height_map, Array2D<int>&& feasibility_info)
    : m_items { std::move(items) },
      m_height_map { std::move(height_map) },
      m_feasibility_info { std::move(feasibility_info) } {}
  
  auto create_feasibility_info(const Item&) const noexcept -> Array2D<int>;

  std::vector<Item> m_items;
  Array2D<int> m_height_map;
  Array2D<int> m_feasibility_info;
};