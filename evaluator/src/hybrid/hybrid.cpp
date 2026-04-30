#include "hh/hybrid/hybrid.hpp"
#include <algorithm>
#include <cmath>

namespace hh {

HybridSS::HybridSS(std::size_t q_e, std::size_t q_a)
  : head_(q_e), tail_(std::make_unique<SpaceSaving>(std::max<std::size_t>(q_a,1))) {
  // tail admission callback will be wired when/if user sets on_admit_
}

void HybridSS::reconfigure(std::size_t q_e, std::size_t q_a, std::size_t n_param, double head_mass_frac) {
  head_.set_capacity(q_e);
  const double p_e = (head_mass_frac < 0.0)
      ? head_mass_fraction()
      : std::clamp(head_mass_frac, 0.0, 1.0);
  const std::size_t residual_floor = (n_param == 0)
      ? 1
      : std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(static_cast<double>(n_param) * (1.0 - p_e))));
  const std::size_t q_tail = std::max<std::size_t>(std::max<std::size_t>(q_a, residual_floor), 1);

  // rebuild tail to enforce capacity precisely
  auto new_tail = std::make_unique<SpaceSaving>(q_tail);
  if (on_admit_) new_tail->set_admission_callback(on_admit_);
  tail_.swap(new_tail);
}

void HybridSS::update(const Id128& id, int w) {
  if (w <= 0) return;
  N_local_ += static_cast<std::uint64_t>(w);

  if (head_.contains(id)) {
    head_.admit_and_add(id, static_cast<std::uint32_t>(w));
    return;
  }

  // Not in head: send to tail (head membership is pre-seeded by caller)
  tail_->update(id, w);
}

Snapshot HybridSS::snapshot() const {
  Snapshot s{};
  s.N_local = N_local_;
  // head first (exact)
  for (const auto& kv : head_.items()) s.candidates.push_back({kv.first, kv.second});
  // then tail
  Snapshot ts = tail_->snapshot();
  for (const auto& c : ts.candidates) s.candidates.push_back(c);
  return s;
}

SnapshotEx HybridSS::snapshot_ex() const {
  SnapshotEx sx{};
  sx.N_local = N_local_;
  sx.q_local = tail_ ? tail_->capacity() : 0;
  sx.head_mass = head_mass();
  sx.head_size = head_size();
  // head: exact => lb=est=count
  sx.candidates.reserve(head_.size());
  sx.errors.reserve(head_.size());
  for (const auto& kv : head_.items()) {
    sx.candidates.push_back({kv.first, kv.second});
    sx.errors.push_back({kv.second, 0});
  }
  // tail: whatever SS provides (lb typically 0)
  SnapshotEx tx = tail_->snapshot_ex();
  if (tx.has_error_bounds()) {
    for (std::size_t i = 0; i < tx.candidates.size(); ++i) {
      sx.candidates.push_back(tx.candidates[i]);
      sx.errors.push_back(tx.errors[i]);
    }
  } else {
    for (const auto& c : tx.candidates) {
      sx.candidates.push_back(c);
      sx.errors.push_back({0, 0});
    }
  }
  return sx;
}

void HybridSS::reset_window() {
  N_local_ = 0;
  head_.clear();
  tail_->reset_window();
}

std::size_t HybridSS::memory_bytes() const {
  std::size_t bytes = sizeof(*this);
  const auto& hm = head_.items();
  bytes += hm.bucket_count() * sizeof(void*); // hash table buckets
  bytes += hm.size() * sizeof(std::pair<const Id128, std::uint32_t>);
  if (tail_) bytes += tail_->memory_bytes();
  return bytes;
}

void HybridSS::seed_head(const std::vector<Id128>& ids) {
  head_.clear();
  std::size_t admitted = 0;
  for (const auto& id : ids) {
    if (admitted >= head_.capacity()) break;
    if (head_.admit_and_add(id, 0)) ++admitted;
  }
}

std::size_t HybridSS::head_size() const { return head_.size(); }
std::size_t HybridSS::head_capacity() const { return head_.capacity(); }

std::uint64_t HybridSS::head_mass() const {
  std::uint64_t sum = 0;
  for (const auto& kv : head_.items()) sum += kv.second;
  return sum;
}

double HybridSS::head_mass_fraction() const {
  if (N_local_ == 0) return 0.0;
  return static_cast<double>(head_mass()) / static_cast<double>(N_local_);
}

} // namespace hh
