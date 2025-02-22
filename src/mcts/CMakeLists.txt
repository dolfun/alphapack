include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4
)
FetchContent_MakeAvailable(pybind11)

add_library(
  mcts
  array2d.h
  item.h
  state.h
  state.cc
  inference_queue.h
  inference_queue.cc
  mcts.h
  mcts.cc
  evaluation.h
  generate_init_states.h
  generate_init_states.cc
  generate_episode.h
  generate_episode.cc
)

target_compile_features(mcts PRIVATE cxx_std_23)
target_link_libraries(mcts PRIVATE pybind11::pybind11)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (MSVC)
    message("Enabling optimizations for MSVC")
    target_compile_options(mcts PUBLIC /fp:fast /GL)

  elseif (CMAKE_C_COMPILER_ID MATCHES "GNU")
    message("Enabling optimizations for GCC")
    target_compile_options(mcts PUBLIC -O3 -ffast-math)

  elseif (CMAKE_C_COMPILER_ID MATCHES "Clang")
    message("Enabling optimizations for Clang")
    target_compile_options(mcts PUBLIC -O3 -ffast-math)
  endif()
endif()