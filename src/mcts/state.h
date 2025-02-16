#pragma once
#include <string>
#include <cstdint>
#include "item.h"
#include "array2d.h"

class State {
public:
  State(const std::vector<Item>& items)
    : m_items { items },
      m_height_map { State::bin_length, State::bin_length },
      m_feasibility_info { create_feasibility_info(m_items.front()) } {}

  auto items() const noexcept -> const std::vector<Item>&;
  auto height_map() const noexcept -> const Array2D<int8_t>&;
  auto feasibility_mask() const noexcept -> Array2D<int8_t>;
  auto normalized_items() const noexcept -> std::vector<float>;
  float packing_efficiency() const noexcept;

  auto possible_actions() const -> std::vector<int>;
  [[nodiscard]] float transition(int);

  static std::string serialize(const State&);
  static State unserialize(const std::string&);

  static constexpr int bin_length = 10;
  static constexpr int bin_height = 10;
  static constexpr int action_count = bin_length * bin_length;
  static constexpr int item_count = 48;
  static constexpr int values_per_item = 3;

private:
  State(std::vector<Item>&& items, Array2D<int8_t>&& height_map, Array2D<int8_t>&& feasibility_info)
    : m_items { std::move(items) },
      m_height_map { std::move(height_map) },
      m_feasibility_info { std::move(feasibility_info) } {}

  friend auto get_state_symmetry(const State&, int) noexcept -> State;

  auto create_feasibility_info(const Item&) const noexcept -> Array2D<int8_t>;

  std::vector<Item> m_items;
  Array2D<int8_t> m_height_map;
  Array2D<int8_t> m_feasibility_info;
};

auto get_state_symmetry(const State&, int) noexcept -> State;
auto get_inverse_priors_symmetry(const State&, const std::vector<float>&, int) noexcept -> std::vector<float>;