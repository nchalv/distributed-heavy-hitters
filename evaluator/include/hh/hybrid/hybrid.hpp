#pragma once
#include <memory>
#include <cstdint>
#include <string>
#include <vector>
#include "hh/sketches/isketch.hpp"
#include "hh/sketches/ss.hpp"
#include "hh/exact/exact_head.hpp"

namespace hh {

// Hybrid sketch: exact-counting head (capacity q_e, pre-seeded) + SpaceSaving tail (capacity q_a).
// Within a window:
//  - If id is in head: update exactly (no error).
//  - Else: send to tail (approx).
//
// Caller is responsible for seeding the head at window start (e.g., from previous certified set).
class HybridSS : public ISketch {
public:
  HybridSS(std::size_t q_e, std::size_t q_a);

  // ISketch interface
  void update(const Id128& id, int w) override;
  Snapshot snapshot() const override;
  SnapshotEx snapshot_ex() const override;
  void reset_window() override;
  std::size_t memory_bytes() const override;

  void set_admission_callback(AdmitFn cb) override {
    on_admit_ = std::move(cb);
    if (tail_) tail_->set_admission_callback(on_admit_);
  }

  // Backward-compatible override (treat single-arg as tail resize).
  void reconfigure(std::size_t new_capacity) override {
    reconfigure(head_capacity(), new_capacity, /*n_param=*/1, /*head_mass_frac=*/head_mass_fraction());
  }
  // Reconfigure sizes between windows. tail capacity is floored by n_param*(1 - head_mass_frac) for discoverability.
  // If head_mass_frac < 0, the current head_mass_fraction() is used.
  void reconfigure(std::size_t q_e, std::size_t q_a, std::size_t n_param = 1, double head_mass_frac = -1.0);

  // Seed head membership for the coming window (counts start at 0). Excess ids beyond capacity are ignored.
  void seed_head(const std::vector<Id128>& ids);

  // Telemetry
  std::size_t head_size() const;
  std::size_t head_capacity() const;
  std::uint64_t head_mass() const;        // sum of counts in head
  double head_mass_fraction() const;      // head_mass / N_local (0 if N_local==0)

private:
  ExactHead head_;
  std::unique_ptr<SpaceSaving> tail_;
  std::uint64_t N_local_{0};
  AdmitFn on_admit_{};
};

} // namespace hh
