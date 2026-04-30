#include "hh/oracle/oracle_all.hpp"

namespace hh {

void OracleAll::update(const Id128& id, int weight) {
  N_ += (weight > 0 ? weight : 0);
  auto it = cnt_.find(id);
  if (it == cnt_.end()) {
    cnt_.emplace(id, (weight > 0 ? (std::uint64_t)weight : 0));
    if (on_admit_) on_admit_(id);   // first time we see this id in the window
  } else {
    if (weight > 0) it->second += (std::uint64_t)weight;
  }
}

Snapshot OracleAll::snapshot() const {
  Snapshot s;
  s.N_local = N_;
  s.candidates.reserve(cnt_.size());
  for (const auto& kv : cnt_) {
    s.candidates.push_back(Cand{kv.first, static_cast<std::uint32_t>(kv.second)});
  }
  return s;
}

SnapshotEx OracleAll::snapshot_ex() const {
  SnapshotEx x;
  x.N_local = N_;
  x.q_local = 0;
  x.head_mass = 0;
  x.head_size = 0;
  x.candidates.reserve(cnt_.size());
  x.errors.reserve(cnt_.size());
  for (const auto& kv : cnt_) {
    const auto est = static_cast<std::uint32_t>(kv.second);
    x.candidates.push_back(Cand{kv.first, est});
    x.errors.push_back(CandErr{est, 0}); // oracle: lb == est, eps=0
  }
  return x;
}

void OracleAll::reset_window() {
  cnt_.clear();
  N_ = 0;
}

} // namespace hh
