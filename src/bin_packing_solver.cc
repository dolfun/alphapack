#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "mcts/generate_init_states.h"
#include "mcts/generate_episode.h"
#include "mcts/training_sample.h"
namespace py = pybind11;

template <typename T>
py::buffer_info array_2d_buffer_info(State::Array2D<T>& arr) {
  py::buffer_info info{};
  info.ptr = static_cast<void*>(arr.data());
  info.itemsize = sizeof(T);
  info.format = py::format_descriptor<T>::format();
  info.ndim = 2;

  info.shape = {
    static_cast<long>(arr.template size<0>()),
    static_cast<long>(arr.template size<1>())
  };

  info.strides = {
    static_cast<long>(sizeof(T) * arr.template size<1>()),
    sizeof(T)
  };

  return info;
}

template <typename T, size_t C>
py::buffer_info array_3d_buffer_info(State::Array3D<T, C>& arr) {
  py::buffer_info info{};
  info.ptr = static_cast<void*>(arr.data());
  info.itemsize = sizeof(T);
  info.format = py::format_descriptor<T>::format();
  info.ndim = 3;

  info.shape = {
    static_cast<long>(arr.template size<0>()),
    static_cast<long>(arr.template size<1>()),
    static_cast<long>(arr.template size<2>())
  };

  info.strides = {
    static_cast<long>(sizeof(T) * arr.template size<1>() * arr.template size<2>()),
    static_cast<long>(sizeof(T) * arr.template size<2>()),
    sizeof(T)
  };

  return info;
}

PYBIND11_MODULE(bin_packing_solver, m) {
  // glm::vec3
  py::class_<Vec3i>(m, "Vec3i")
    .def(py::init<int, int, int>(), py::arg("x"), py::arg("y"), py::arg("z"))
    .def_readwrite("x", &Vec3i::x)
    .def_readwrite("y", &Vec3i::y)
    .def_readwrite("z", &Vec3i::z);

  // Item
  py::class_<Item>(m, "Item")
    .def(py::init<>())
    .def_readwrite("shape", &Item::shape)
    .def_readwrite("placed", &Item::placed);

  // Array2D
  py::class_<State::Array2D<int8_t>>(m, "Array2Di", py::buffer_protocol())
    .def_buffer(array_2d_buffer_info<int8_t>);

  // Array3D
  py::class_<State::Array3D<float, State::input_feature_count>>(m, "ImageData", py::buffer_protocol())
    .def_buffer(array_3d_buffer_info<float, State::input_feature_count>);

  // State
  py::class_<State>(m, "State")
    .def(py::init<std::vector<Item>>(), py::arg("items"))

    .def_property_readonly("items", &State::items)
    .def_property_readonly("height_map", &State::height_map)
    .def_property_readonly("feasibility_mask", &State::feasibility_mask)
    .def_property_readonly("packing_efficiency", &State::packing_efficiency)
    .def("inference_input", &State::inference_input, py::arg("k"))

    .def_property_readonly("possible_actions", &State::possible_actions)
    .def("transition", &State::transition, py::arg("action_idx"))

    .def_readonly_static("bin_length", &State::bin_length)
    .def_readonly_static("bin_height", &State::bin_height)
    .def_readonly_static("action_count", &State::action_count)
    .def_readonly_static("item_count", &State::item_count)
    .def_readonly_static("input_feature_count", &State::input_feature_count)
    .def_readonly_static("additional_input_count", &State::additional_input_count)
    .def_readonly_static("value_support_count", &State::value_support_count)

    .def(py::pickle(
      [] (const State& state) { return py::bytes(State::serialize(state)); },
      [] (const py::bytes& bytes) { return State::unserialize(bytes); }
    ));

  // InferInput
  using InferInput = State::InferInput;
  py::class_<InferInput, std::shared_ptr<InferInput>>(m, "InferInput")
    .def_readonly("image_data", &InferInput::image_data)
    .def_readonly("additional_data", &InferInput::additional_data);

  // Evaluation
  using mcts::Evaluation;
  py::class_<Evaluation>(m, "Evaluation")
    .def_readwrite("state", &Evaluation::state)
    .def_readwrite("action_idx", &Evaluation::action_idx)
    .def_readwrite("priors", &Evaluation::priors)
    .def_readwrite("value", &Evaluation::value)
    .def(py::pickle(
      [] (const Evaluation& evaluation) {
        return py::make_tuple(evaluation.state, evaluation.action_idx, evaluation.priors, evaluation.value);
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

  // generate_states
  m.def(
    "generate_random_init_states",
    &generate_random_init_states,
    py::arg("seed"),
    py::arg("pool_size"),
    py::arg("min_item_dim"),
    py::arg("max_item_dim")
  );

  m.def(
    "generate_cut_init_states",
    &generate_cut_init_states,
    py::arg("seed"),
    py::arg("pool_size"),
    py::arg("min_item_dim"),
    py::arg("max_item_dim"),
    py::arg("min_packing_efficiency"),
    py::arg("max_packing_efficiency"),
    py::arg("count")
  );

  // generate_episodes
  m.def(
    "generate_episodes",
    &mcts::generate_episodes,
    py::arg("states"),
    py::arg("episodes_count"),
    py::arg("worker_count"),
    py::arg("move_threshold"),
    py::arg("simulations_per_move"),
    py::arg("mcts_thread_count"),
    py::arg("c_puct"),
    py::arg("virtual_loss"),
    py::arg("alpha"),
    py::arg("batch_size"),
    py::arg("infer_func")
  );

  // generate_samples
  py::class_<TrainingSample>(m, "TrainingSample")
    .def_readonly("input", &TrainingSample::input)
    .def_readonly("priors", &TrainingSample::priors)
    .def_readonly("value", &TrainingSample::value);

  m.def(
    "prepare_samples", 
    &prepare_training_samples,
    py::arg("episodes")
  );
}