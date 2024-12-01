#include "package.h"

glm::ivec3 Package::get_shape_along_axes(glm::ivec3 shape, int orientation) {
  glm::ivec3 shape_along_axes;
  for (int i = 0; i < 3; ++i) {
    shape_along_axes[i] = shape[ORIENTATIONS[orientation][i]];
  }
  return shape_along_axes;
}