#pragma once
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include "hh/core/id128.hpp"

namespace hh {

// Bounded exact counter for the hybrid head.
// Admits up to cap_ distinct keys per window; extra keys are not admitted.
class ExactHead {
public:
  explicit ExactHead(std::size_t cap = 0) : cap_(cap) {
    map_.max_load_factor(kMaxLoadFactor);
    reserve_for_cap(cap_);
  }

  void set_capacity(std::size_t cap) {
    cap_ = cap;
    reserve_for_cap(cap_);
  }
  std::size_t capacity() const { return cap_; }
  std::size_t size() const { return map_.size(); }

  bool contains(const Id128& id) const { return map_.find(id) != map_.end(); }

  // Returns true if admitted (present or newly inserted), false if rejected (at cap).
  bool admit_and_add(const Id128& id, std::uint32_t w) {
    auto it = map_.find(id);
    if (it != map_.end()) { it->second += w; return true; }
    if (map_.size() >= cap_) return false;
    map_.emplace(id, w);
    return true;
  }

  std::uint32_t get(const Id128& id) const {
    auto it = map_.find(id);
    return (it == map_.end()) ? 0u : it->second;
  }

  const std::unordered_map<Id128, std::uint32_t, Id128Hash>& items() const { return map_; }

  void clear() { map_.clear(); }

private:
  static constexpr float kMaxLoadFactor = 2.0f;

  void reserve_for_cap(std::size_t cap) {
    if (cap == 0) return;
    const auto buckets = static_cast<std::size_t>(
        std::ceil(static_cast<double>(cap) / static_cast<double>(kMaxLoadFactor)));
    if (map_.bucket_count() < buckets) map_.rehash(std::max<std::size_t>(buckets, 1));
  }

  std::size_t cap_;
  std::unordered_map<Id128, std::uint32_t, Id128Hash> map_;
};

} // namespace hh
