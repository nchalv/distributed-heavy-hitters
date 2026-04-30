#pragma once
#include "hh/sketches/isketch.hpp"
#include <unordered_map>

namespace hh {

// Oracle sketch: exact counting of all items in the window (reference method).
class OracleAll : public ISketch {
public:
  OracleAll() = default;
  ~OracleAll() override = default;

  void update(const Id128& id, int weight = 1) override;
  Snapshot snapshot() const override;
  SnapshotEx snapshot_ex() const override;  // lb = est (exact)
  void reset_window() override;

  void set_admission_callback(ISketch::AdmitFn cb) override { on_admit_ = std::move(cb); }

private:
  std::unordered_map<Id128, std::uint64_t, Id128Hash> cnt_;
  std::uint64_t N_{0};
  ISketch::AdmitFn on_admit_{};
};

} // namespace hh
