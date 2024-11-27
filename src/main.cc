#include <fmt/core.h>
#include <glm/glm.hpp>

int main() {
  glm::ivec3 a { 1, 2, 3 };
  fmt::println("Hello World! {} {} {}", a.x, a.y, a.z);
}