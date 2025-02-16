#include "generate_init_states.h"
#include <random>

auto generate_random_init_states(uint32_t seed, size_t count, int min_item_dim, int max_item_dim)
    -> std::vector<State> {

  std::mt19937 engine { seed };
  std::vector<State> states;
  states.reserve(count);
  for (int i = 0; i < count; ++i) {
    std::uniform_int_distribution<int> dist { min_item_dim, max_item_dim };
    std::vector<Item> items(State::item_count);
    for (auto& item : items) {
      item.shape.x = dist(engine);
      item.shape.y = dist(engine);
      item.shape.z = dist(engine);
    }
    states.emplace_back(items);
  }

  return states;
}

struct PlacedItem {
  Vec3i shape, pos;
};

class CutStateGenerator {
public:
  CutStateGenerator(
    uint32_t seed,
    int _min_item_dim,
    int _max_item_dim,
    float _min_packing_efficiency,
    float _max_packing_efficiency)

    : engine { seed }, 
      min_item_dim { _min_item_dim },
      max_item_dim { _max_item_dim },
      min_packing_efficiency { _min_packing_efficiency },
      max_packing_efficiency { _max_packing_efficiency } {}

  auto generate(size_t count) -> std::vector<State> {
    std::vector<State> states;
    states.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      states.emplace_back(generate_state());
    }
    return states;
  }

private:
  bool is_valid_item(PlacedItem item) {
    return item.shape.x >= min_item_dim && item.shape.x <= max_item_dim &&
           item.shape.y >= min_item_dim && item.shape.y <= max_item_dim &&
           item.shape.z >= min_item_dim && item.shape.z <= max_item_dim;
  }

  auto split_along_axis(PlacedItem item, int axis) -> std::pair<PlacedItem, PlacedItem> {
    auto get_value = [axis] (Vec3i& v) -> int& {
      return reinterpret_cast<int*>(&v)[axis]; // Don't do this
    };

    std::uniform_int_distribution<int> dist { 
      min_item_dim,
      get_value(item.shape) - min_item_dim
    };

    PlacedItem item1 = item, item2 = item;
    int split_point = dist(engine);
    get_value(item1.shape) = split_point;
    get_value(item2.shape) = get_value(item.shape) - split_point;
    get_value(item2.pos) += split_point;
    return { item1, item2 };
  }

  auto generate_state() -> State {
    std::vector<PlacedItem> invalid_items, valid_items;

    invalid_items.emplace_back(
      Vec3i { State::bin_length, State::bin_length, State::bin_height }, 
      Vec3i { 0, 0, 0 }
    );

    while (!invalid_items.empty()) {
      std::uniform_int_distribution<size_t> invalid_item_dist { 0, invalid_items.size() - 1 };
      auto it = invalid_items.begin() + invalid_item_dist(engine);
      PlacedItem item = *it;
      invalid_items.erase(it);

      std::vector<int> valid_axes;
      if (item.shape.x >= 2 * min_item_dim) valid_axes.push_back(0);
      if (item.shape.y >= 2 * min_item_dim) valid_axes.push_back(1);
      if (item.shape.z >= 2 * min_item_dim) valid_axes.push_back(2);

      std::uniform_int_distribution<size_t> axes_dist { 0, valid_axes.size() - 1 };
      int axis = valid_axes[axes_dist(engine)];
      auto [item1, item2] = split_along_axis(item, axis);

      auto insert_item = [&] (PlacedItem item) {
        if (is_valid_item(item)) {
          valid_items.emplace_back(item);
        } else {
          invalid_items.emplace_back(item);
        }
      };
      insert_item(item1);
      insert_item(item2);
    }

    std::ranges::sort(valid_items, {}, [] (PlacedItem item) {
      return std::make_tuple(item.pos.z, item.pos.x, item.pos.y);
    });

    Item default_item {
      .shape = {},
      .placed = true
    };

    std::vector<Item> items(State::item_count, default_item);
    for (size_t i = 0; i < std::min(items.size(), valid_items.size()); ++i) {
      items[i].shape = valid_items[i].shape;
      items[i].placed = false;
    }

    static std::random_device rd{};
    static std::mt19937 engine { rd() };
    std::uniform_real_distribution<float> dist { min_packing_efficiency, max_packing_efficiency };
    float expected_packing_efficiency = dist(engine);

    State state { items };
    for (auto item : valid_items) {
      if (state.packing_efficiency() >= expected_packing_efficiency) break;
      if (state.possible_actions().empty()) break;

      int action_idx = item.pos.x * State::bin_length + item.pos.y;
      (void)state.transition(action_idx);
    }

    return state;
  }

  std::mt19937 engine;
  int min_item_dim, max_item_dim;
  float min_packing_efficiency, max_packing_efficiency;
};

auto generate_cut_init_states(
  std::uint32_t seed,
  size_t count,
  int min_item_dim,
  int max_item_dim,
  float min_packing_efficiency,
  float max_packing_efficiency)
    -> std::vector<State> {

  CutStateGenerator generator { seed, min_item_dim, max_item_dim, min_packing_efficiency, max_packing_efficiency };
  return generator.generate(count);
}