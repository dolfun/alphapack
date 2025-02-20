#pragma once

struct Vec3i {
  int x, y, z;

  // Don't do this
  int operator[] (int idx) const noexcept {
    return reinterpret_cast<const int*>(this)[idx];
  }

  int& operator[] (int idx) noexcept {
    return reinterpret_cast<int*>(this)[idx];
  }
};

struct Item {
  Vec3i shape;
  bool placed;
};