#pragma once
#include "hh/sketches/isketch.hpp"
#include "hh/core/hash.hpp"
#include "hh/sketches/lossy_strategy.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>
#include <cstring>

namespace hh {

struct HLBucketCand {
  Id128 id{};
  std::uint32_t est{0};
  std::uint32_t bucket{0};
};

struct HLBucketSnapshot {
  std::uint64_t N_local{0};
  std::size_t w{0};
  std::size_t d{0};
  std::vector<HLBucketCand> candidates;
};

// Single-thread HeavyLocker with LossyStrategy admission.
// - 1 hash → 1 bucket (width = w_)
// - d_ slots per bucket, lightly ordered by count (single swap per update)
// - bucket full → apply Lossy strategy to smallest slot iff
//   smallest < N_local * (theta_phi_ * lock_L_)
class HeavyLocker : public ISketch {
public:
  // lossy_mode: 0=MinusOne, 1=HeavyKeeper, 2=RAP (default), 3=USS)
  HeavyLocker(std::size_t w, std::size_t d, double lock_L, double theta_phi, int lossy_mode = 2);

  void update(const Id128& id, int weight) override;
  Snapshot snapshot() const override;
  SnapshotEx snapshot_ex() const override;
  HLBucketSnapshot snapshot_bucketed() const;
  void reset_window() override;
  std::size_t memory_bytes() const override;
  void set_admission_callback(AdmitFn cb) override { on_admit_ = std::move(cb); }

private:
  struct Slot {
    Id128 id{};
    std::uint32_t cnt{0};
    bool used{false};
  };
  struct Bucket {
    std::vector<Slot> slots; // size d_
    bool lock_bit{false};
  };

  std::size_t w_{0};
  std::size_t d_{0};
  double lock_L_{0.7};       // L from the paper (default 0.7)
  double theta_phi_{0.01};   // theta = 1/n_param (set by caller)
  std::uint64_t N_local_{0};
  std::vector<Bucket> table_;

  Lossy::Context lossy_;
  AdmitFn on_admit_{};

  std::size_t index_of(const Id128& id) const;
  static void bubble_up(std::vector<Slot>& slots, std::size_t j);

  static std::string id128_to_bytes(const Id128& id) {
    return std::string(reinterpret_cast<const char*>(id.b.data()), 16);
  }
  static Id128 bytes_to_id128(const std::string& bytes) {
    Id128 id{};
    if (!bytes.empty()) {
      const std::size_t n = std::min<std::size_t>(bytes.size(), id.b.size());
      std::memcpy(id.b.data(), bytes.data(), n);
    }
    return id;
  }
};

} // namespace hh
