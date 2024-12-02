add_library(container_solver 
  array2d.h
  package.h
  package.cc
  package_generator.h
  package_generator.cc
  container.h
  container.cc
)
target_compile_features(container_solver PUBLIC cxx_std_23)
target_link_libraries(container_solver PUBLIC glm::glm)

add_executable(container_solver_train train.cc)
target_link_libraries(container_solver_train PRIVATE container_solver fmt::fmt)