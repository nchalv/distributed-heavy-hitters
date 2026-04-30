#pragma once
#include "hh/core/id128.hpp"
#include <cstdint>
#include <vector>
#include <functional>

namespace hh {

// Basic candidate
struct Cand {
  Id128 id;
  std::uint32_t est;
};

struct Snapshot {
  std::uint64_t N_local{0};
  std::vector<Cand> candidates;
};

// Optional per-candidate error metadata.
// - SpaceSaving: lb = max(0, est - eps), eps = epsilon_i
// - Other sketches may omit this entirely to avoid extra per-candidate storage.
struct CandErr {
  std::uint32_t lb;
  std::uint32_t eps;
};

struct SnapshotEx {
  std::uint64_t N_local{0};
  std::vector<Cand> candidates;   // always present
  std::vector<CandErr> errors;    // optional; if present, size == candidates.size()
  std::size_t q_local{0};       // capacity (entries) if applicable (SS), else 0
  // Optional hybrid telemetry (head mass/size); non-hybrid sketches leave at 0.
  std::uint64_t head_mass{0};
  std::size_t   head_size{0};

  bool has_error_bounds() const { return !errors.empty() && errors.size() == candidates.size(); }
  std::uint32_t cand_lb(std::size_t i) const {
    return has_error_bounds() ? errors[i].lb : 0u;
  }
  std::uint32_t cand_eps(std::size_t i) const {
    return has_error_bounds() ? errors[i].eps : 0u;
  }
};

class ISketch {
public:
  using AdmitFn = std::function<void(const Id128&)>;

  virtual ~ISketch() = default;
  virtual void update(const Id128& id, int weight = 1) = 0;
  virtual Snapshot snapshot() const = 0;
  virtual void reset_window() = 0;

  // Rich snapshot; default builds from snapshot() with lb=0.
  virtual SnapshotEx snapshot_ex() const {
    Snapshot s = snapshot();
    SnapshotEx x; x.N_local = s.N_local; x.q_local = 0; x.candidates = std::move(s.candidates);
    return x;
  }

  // Admission callback (fires when an unmonitored id becomes resident). Default: noop.
  virtual void set_admission_callback(AdmitFn) {}
  virtual void reconfigure(std::size_t /*new_capacity*/) {}
  // Best-effort memory usage of sketch state (bytes). Implementations may return 0 if unknown.
  virtual std::size_t memory_bytes() const { return 0; }
};

} // namespace hh
