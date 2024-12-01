#pragma once
#include <glm/vec3.hpp>

enum class PackageType {
  priority,
  economy
};

struct Package {
  glm::ivec3 shape;
  int weight;
  PackageType type;
  int cost;

  bool is_placed;
  glm::ivec3 pos;

  static constexpr int ORIENTATIONS[6][3] = {
    { 0, 1, 2 },
    { 0, 2, 1 },
    { 1, 0, 2 },
    { 1, 2, 0 },
    { 2, 0, 1 },
    { 2, 1, 0 },
  };

  static glm::ivec3 get_shape_along_axes(glm::ivec3, int);
};