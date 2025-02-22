#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "item.h"
#include "array2d.h"

class State {
public:
  static constexpr int bin_length = 10;
  static constexpr int bin_height = 10;
  static constexpr int action_count = bin_length * bin_length;
  static constexpr int item_count = 64;
  static constexpr int values_per_item = 4;
  static constexpr int value_support_count = 101;

  template <typename T>
  using Array2D = Array2D<T, bin_length, bin_height>;

  State(const std::vector<Item>&);

  auto items() const noexcept -> const std::array<Item, item_count>&;
  auto height_map() const noexcept -> const Array2D<int8_t>&;
  auto feasibility_mask() const noexcept -> Array2D<int8_t>;
  auto normalized_items() const noexcept -> std::vector<float>;
  float packing_efficiency() const noexcept;

  auto possible_actions() const -> std::vector<int>;
  [[nodiscard]] float transition(int);

  static std::string serialize(const State&);
  static State unserialize(const std::string&);

private:
  State(std::array<Item, item_count>&&, Array2D<int8_t>&&, Array2D<int8_t>&&);

  friend auto get_state_symmetry(const State&, int) noexcept -> State;

  auto create_feasibility_info(const Item&) const noexcept -> Array2D<int8_t>;

  std::array<Item, item_count> m_items;
  Array2D<int8_t> m_height_map;
  Array2D<int8_t> m_feasibility_info;
};

auto get_state_symmetry(const State&, int) noexcept -> State;
auto get_inverse_priors_symmetry(const State&, const std::array<float, State::action_count>&, int) noexcept
  -> std::array<float, State::action_count>;