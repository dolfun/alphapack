include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 0c9fce2ffefecfdce794e1859584e25877b7b592
)
FetchContent_MakeAvailable(fmt)

add_subdirectory(container-solver)

add_executable(main main.cc)
target_compile_features(main PRIVATE cxx_std_23)
target_link_libraries(main PRIVATE fmt::fmt)