#pragma once

struct Vec3i {
  int x, y, z;
};

struct Item {
  Vec3i shape;
  bool placed;
};