#pragma once

struct Vec3i {
  union {
    int x, y, z;
    int data[3];
  };

  constexpr int& operator[] (int idx) noexcept {
    return data[idx];
  }

  constexpr const int& operator[] (int idx) const noexcept {
    return data[idx];
  }
};

struct Item {
  Vec3i shape, pos;
  bool placed;
};