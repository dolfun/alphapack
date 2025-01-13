#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "mcts/state.h"
#include "mcts/generate_episode.h"
namespace py = pybind11;

template <typename T>
py::buffer_info array_2d_buffer_info(Array2D<T>& arr) {
  py::buffer_info info{};
  info.ptr = static_cast<void*>(arr.data());
  info.itemsize = sizeof(T);
  info.format = py::format_descriptor<T>::format();
  info.ndim = 2;
  info.shape = { static_cast<long>(arr.rows()), static_cast<long>(arr.cols()) };
  info.strides = { static_cast<long>(sizeof(T) * arr.cols()), sizeof(T) };
  return info;
}

PYBIND11_MODULE(bin_packing_solver, m) {
  // glm::vec3
  py::class_<Vec3i>(m, "Vec3i")
    .def_readwrite("x", &Vec3i::x)
    .def_readwrite("y", &Vec3i::y)
    .def_readwrite("z", &Vec3i::z);

  // Item
  py::class_<Item>(m, "Item")
    .def(py::init<>())
    .def_readwrite("shape", &Item::shape)
    .def_readwrite("placed", &Item::placed)
    .def_readwrite("pos", &Item::pos);

  // Array2D<int>
  py::class_<Array2D<int>>(m, "Array2Di", py::buffer_protocol())
    .def_buffer(array_2d_buffer_info<int>);
  
  // State
  py::class_<State, std::shared_ptr<State>>(m, "State")
    .def(py::init<std::vector<Item>>(), py::arg("items"))

    .def_property_readonly("items", &State::items)
    .def_property_readonly("height_map", &State::height_map)

    .def_property_readonly("normalized_items", &State::normalized_items)

    .def_property_readonly("possible_actions", &State::possible_actions)
    .def("transition", &State::transition, py::arg("action_idx"))
    .def_property_readonly("reward", &State::reward)

    .def("serialize", &State::serialize)
    .def("unserialize", &State::unserialize)

    .def_readonly_static("bin_length", &State::bin_length)
    .def_readonly_static("bin_height", &State::bin_height)
    .def_readonly_static("action_count", &State::action_count)
    .def_readonly_static("item_count", &State::item_count)
    .def_readonly_static("values_per_item", &State::values_per_item)

    .def(py::pickle(
      [] (const State& state) { return state.serialize(); }, 
      &State::unserialize
    ));

  // Evaluation
  using mcts::Evaluation;
  py::class_<Evaluation>(m, "Evaluation")
    .def_readwrite("state", &Evaluation::state)
    .def_readwrite("action_idx", &Evaluation::action_idx)
    .def_readwrite("priors", &Evaluation::priors)
    .def_readwrite("reward", &Evaluation::reward)
    .def(py::pickle(
      [] (const Evaluation& evaluation) {
        return py::make_tuple(evaluation.state, evaluation.action_idx, evaluation.priors, evaluation.reward);
      },
      [] (py::tuple t) {
        Evaluation evaluation {
          t[0].cast<State>(),
          t[1].cast<int>(),
          t[2].cast<std::vector<float>>(),
          t[3].cast<float>()
        };
        return evaluation;
      }
    ));

  // generate episode
  m.def(
    "generate_episodes",
    &mcts::generate_episodes,
    py::arg("seed"),
    py::arg("seed_pool_size"),
    py::arg("episodes_count"),
    py::arg("worker_count"),
    py::arg("simulations_per_move"),
    py::arg("mcts_thread_count"),
    py::arg("batch_size"),
    py::arg("c_puct"),
    py::arg("virtual_loss"),
    py::arg("infer_func")
  );
}