if (WIN32)
  set(BUILD_SHARED_LIBS OFF)
endif()

add_subdirectory(container-solver)

add_executable(main main.cc)
target_compile_features(main PRIVATE cxx_std_20)