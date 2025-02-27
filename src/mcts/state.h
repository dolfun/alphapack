#pragma once
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include "item.h"
#include "array2d.h"
#include "array3d.h"

class State {
public:
  static constexpr int bin_length = 10;
  static constexpr int bin_height = 10;
  static constexpr int action_count = bin_length * bin_length;
  static constexpr int item_count = 64;
  static constexpr int input_feature_count = 2;
  static constexpr int additional_input_count = 4 * item_count;
  static constexpr int value_support_count = 101;

  static constexpr std::pair<int, int> (*symmetric_transforms[8])(int, int, int, int, int) = {
    [] (int x, int y, int l, int w, int L) { return std::make_pair(x        , y        ); },
    [] (int x, int y, int l, int w, int L) { return std::make_pair(L - y - w, x        ); },
    [] (int x, int y, int l, int w, int L) { return std::make_pair(L - x - l, L - y - w); },
    [] (int x, int y, int l, int w, int L) { return std::make_pair(y        , L - x - l); },
    [] (int x, int y, int l, int w, int L) { return std::make_pair(L - x - l, y        ); },
    [] (int x, int y, int l, int w, int L) { return std::make_pair(L - y - w, L - x - l); },
    [] (int x, int y, int l, int w, int L) { return std::make_pair(x        , L - y - w); },
    [] (int x, int y, int l, int w, int L) { return std::make_pair(y        , x        ); },
  };

  template <typename T>
  using Array2D = Array2D<T, bin_length, bin_height>;

  template <typename T, size_t C>
  using Array3D = Array3D<T, C, bin_length, bin_length>;

  State(const std::vector<Item>&);

  auto items() const noexcept -> const std::array<Item, item_count>&;
  auto height_map() const noexcept -> const Array2D<int8_t>&;
  auto feasibility_mask() const noexcept -> Array2D<int8_t>;
  float packing_efficiency() const noexcept;

  auto possible_actions() const -> std::vector<int>;
  [[nodiscard]] float transition(int);

  static std::string serialize(const State&);
  static State unserialize(const std::string&);

  struct InferInput {
    Array3D<float, input_feature_count> image_data;
    std::array<float, additional_input_count> additional_data;
  };

  struct InferResult {
    std::array<float, State::action_count> priors;
    std::array<float, State::value_support_count> value;
  };

  auto inference_input(int) const noexcept -> std::shared_ptr<InferInput>;
  auto invert_symmetric_transform(const std::array<float, State::action_count>&, int) const noexcept
    -> std::array<float, State::action_count>;

private:
  State(const std::array<Item, item_count>&, const Array2D<int8_t>&, const Array2D<int8_t>&);
  auto get_additional_data(bool swap) const noexcept -> std::array<float, additional_input_count>;
  auto create_feasibility_info(const Item&) const noexcept -> Array2D<int8_t>;

  std::array<Item, item_count> m_items;
  Array2D<int8_t> m_height_map;
  Array2D<int8_t> m_feasibility_info;
};