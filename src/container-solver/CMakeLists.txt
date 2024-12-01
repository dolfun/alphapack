add_library(container_solver 
  array2d.h
  package.h
  package.cc
  package_generator.h
  package_generator.cc
  container.h
  container.cc
)
target_compile_features(container_solver PRIVATE cxx_std_23)
target_link_libraries(container_solver PUBLIC glm::glm)