if (WIN32)
  set(BUILD_SHARED_LIBS OFF)
endif()

add_subdirectory(mcts)

pybind11_add_module(
  bin_packing_solver
  bin_packing_solver.cc
)

target_compile_features(bin_packing_solver PUBLIC cxx_std_20)
target_link_libraries(bin_packing_solver PUBLIC mcts)

set(MODULE_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/src)
set_target_properties(
  bin_packing_solver
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  ${MODULE_OUTPUT_DIR}
)

set(STUBGEN_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/stubgen")
if(WIN32)
  set(STUBGEN_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/Scripts/stubgen.exe")
endif()

if (EXISTS ${STUBGEN_EXECUTABLE})
  add_custom_command(TARGET bin_packing_solver POST_BUILD
    COMMAND ${STUBGEN_EXECUTABLE} -m bin_packing_solver -o .
    WORKING_DIRECTORY ${MODULE_OUTPUT_DIR}
  )
endif()