#include "hh/sketches/ss.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace hh {

namespace {
constexpr std::uint16_t LOC_NODE_NPOS = 0xFFFFu;
constexpr std::uint16_t CNT16_MAX = UINT16_MAX;
constexpr std::uint16_t BUCKET16_MAX = UINT16_MAX;
constexpr std::uint16_t EPS16_MAX = UINT16_MAX;
constexpr std::uint16_t LOC_STATE_MASK = 0x0003u;
constexpr std::uint16_t LOC_TAG_SHIFT = 2;
constexpr std::uint16_t LOC_TAG_MASK = 0x3FFFu;

inline std::uint16_t loc_state(std::uint16_t meta) {
  return static_cast<std::uint16_t>(meta & LOC_STATE_MASK);
}

inline std::uint16_t loc_set_state(std::uint16_t meta, std::uint16_t st) {
  return static_cast<std::uint16_t>((meta & ~LOC_STATE_MASK) | (st & LOC_STATE_MASK));
}

inline std::uint16_t loc_entry_tag(std::uint16_t meta) {
  return static_cast<std::uint16_t>((meta >> LOC_TAG_SHIFT) & LOC_TAG_MASK);
}

inline std::uint16_t loc_set_tag(std::uint16_t meta, std::uint16_t tag14) {
  const std::uint16_t clean = static_cast<std::uint16_t>(tag14 & LOC_TAG_MASK);
  return static_cast<std::uint16_t>((meta & LOC_STATE_MASK) | (clean << LOC_TAG_SHIFT));
}
} // namespace

std::size_t SpaceSaving::next_pow2(std::size_t x) {
  if (x <= 1) return 1;
  --x;
  for (std::size_t s = 1; s < sizeof(std::size_t) * 8; s <<= 1) x |= (x >> s);
  return x + 1;
}

std::size_t SpaceSaving::loc_hash(const Id128& id) {
  return Id128Hash{}(id);
}

std::uint16_t SpaceSaving::loc_tag(std::size_t h) {
  const std::uint16_t t = static_cast<std::uint16_t>((h ^ (h >> 16) ^ (h >> 32) ^ (h >> 48)) & LOC_TAG_MASK);
  return t ? t : static_cast<std::uint16_t>(1);
}

void SpaceSaving::loc_reset(std::size_t expected_size) {
  const std::size_t cap = std::max<std::size_t>(8, next_pow2(expected_size * 2));
  loc_slots_.assign(cap, {});
  loc_size_ = 0;
  loc_tombs_ = 0;
}

void SpaceSaving::loc_clear() {
  for (auto& e : loc_slots_) e.meta = 0;
  loc_size_ = 0;
  loc_tombs_ = 0;
}

SpaceSaving::idx_t SpaceSaving::loc_find(const Id128& id) const {
  if (loc_slots_.empty()) return NPOS;
  const std::size_t h = loc_hash(id);
  const std::uint16_t tag = loc_tag(h);
  const std::size_t mask = loc_slots_.size() - 1;
  std::size_t pos = h & mask;

  for (;;) {
    const auto& e = loc_slots_[pos];
    const std::uint16_t st = loc_state(e.meta);
    if (st == 0) return NPOS;
    if (st == 1 && loc_entry_tag(e.meta) == tag && e.node < nodes_.size() && nodes_[e.node].id == id) return e.node;
    pos = (pos + 1) & mask;
  }
}

void SpaceSaving::loc_rehash(std::size_t new_capacity) {
  std::vector<LocEntry> old = std::move(loc_slots_);
  loc_slots_.assign(std::max<std::size_t>(8, next_pow2(new_capacity)), {});
  loc_size_ = 0;
  loc_tombs_ = 0;

  const std::size_t mask = loc_slots_.size() - 1;
  for (const auto& e : old) {
    if (loc_state(e.meta) != 1) continue;
    const idx_t node_idx = e.node;
    if (node_idx >= nodes_.size()) continue;
    assert(node_idx < LOC_NODE_NPOS);
    const Id128& key = nodes_[node_idx].id;
    const std::size_t h = loc_hash(key);
    std::size_t pos = h & mask;
    while (loc_state(loc_slots_[pos].meta) == 1) pos = (pos + 1) & mask;
    auto& dst = loc_slots_[pos];
    dst.node = static_cast<std::uint16_t>(node_idx);
    dst.meta = loc_set_tag(dst.meta, loc_tag(h));
    dst.meta = loc_set_state(dst.meta, 1);
    ++loc_size_;
  }
}

void SpaceSaving::loc_insert(const Id128& id, idx_t node_idx) {
  if (loc_slots_.empty()) loc_reset(m_);
  assert(node_idx < LOC_NODE_NPOS);
  if ((loc_size_ + loc_tombs_ + 1) * 10 >= loc_slots_.size() * 7) {
    loc_rehash(loc_slots_.size() * 2);
  }

  const std::size_t h = loc_hash(id);
  const std::uint16_t tag = loc_tag(h);
  const std::size_t mask = loc_slots_.size() - 1;
  std::size_t pos = h & mask;
  std::size_t first_tomb = static_cast<std::size_t>(-1);

  for (;;) {
    auto& e = loc_slots_[pos];
    const std::uint16_t st = loc_state(e.meta);
    if (st == 0) {
      std::size_t ins = (first_tomb == static_cast<std::size_t>(-1)) ? pos : first_tomb;
      auto& dst = loc_slots_[ins];
      dst.node = static_cast<std::uint16_t>(node_idx);
      dst.meta = loc_set_tag(dst.meta, tag);
      if (loc_state(dst.meta) == 2) --loc_tombs_;
      dst.meta = loc_set_state(dst.meta, 1);
      ++loc_size_;
      return;
    }
    if (st == 2) {
      if (first_tomb == static_cast<std::size_t>(-1)) first_tomb = pos;
    } else if (loc_entry_tag(e.meta) == tag && e.node < nodes_.size() && nodes_[e.node].id == id) {
      e.node = static_cast<std::uint16_t>(node_idx);
      e.meta = loc_set_tag(e.meta, tag);
      return;
    }
    pos = (pos + 1) & mask;
  }
}

void SpaceSaving::loc_erase(const Id128& id) {
  if (loc_slots_.empty()) return;
  const std::size_t h = loc_hash(id);
  const std::uint16_t tag = loc_tag(h);
  const std::size_t mask = loc_slots_.size() - 1;
  std::size_t pos = h & mask;

  for (;;) {
    auto& e = loc_slots_[pos];
    const std::uint16_t st = loc_state(e.meta);
    if (st == 0) return;
    if (st == 1 && loc_entry_tag(e.meta) == tag && e.node < nodes_.size() && nodes_[e.node].id == id) {
      e.meta = loc_set_state(e.meta, 2);
      --loc_size_;
      ++loc_tombs_;
      if (loc_tombs_ > loc_slots_.size() / 3) {
        loc_rehash(loc_slots_.size());
      }
      return;
    }
    pos = (pos + 1) & mask;
  }
}

std::uint32_t SpaceSaving::node_count(idx_t node_idx) const {
  return wide_counts_ ? wide_cnts_[node_idx] : static_cast<std::uint32_t>(nodes_[node_idx].cnt);
}

void SpaceSaving::set_node_count(idx_t node_idx, std::uint32_t value) {
  if (wide_counts_) {
    wide_cnts_[node_idx] = value;
  } else {
    nodes_[node_idx].cnt = static_cast<std::uint16_t>(value);
  }
}

void SpaceSaving::promote_counts_to_wide() {
  if (wide_counts_) return;
  wide_cnts_.resize(nodes_.size(), 0);
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    wide_cnts_[i] = static_cast<std::uint32_t>(nodes_[i].cnt);
  }
  wide_counts_ = true;
}

std::uint32_t SpaceSaving::bucket_value(idx_t bucket_idx) const {
  return wide_bucket_values_mode_ ? wide_bucket_vals_[bucket_idx]
                             : static_cast<std::uint32_t>(buckets_[bucket_idx].value);
}

void SpaceSaving::set_bucket_value(idx_t bucket_idx, std::uint32_t value) {
  if (wide_bucket_values_mode_) {
    wide_bucket_vals_[bucket_idx] = value;
  } else {
    buckets_[bucket_idx].value = static_cast<std::uint16_t>(value);
  }
}

void SpaceSaving::promote_bucket_values_to_wide() {
  if (wide_bucket_values_mode_) return;
  wide_bucket_vals_.resize(buckets_.size(), 0);
  for (std::size_t i = 0; i < buckets_.size(); ++i) {
    wide_bucket_vals_[i] = static_cast<std::uint32_t>(buckets_[i].value);
  }
  wide_bucket_values_mode_ = true;
}

std::uint32_t SpaceSaving::item_eps(idx_t node_idx) const {
  if (!per_item_eps_mode_) return eps_max_;
  return wide_per_item_eps_mode_
      ? wide_per_item_eps_[node_idx]
      : static_cast<std::uint32_t>(per_item_eps16_[node_idx]);
}

void SpaceSaving::set_item_eps(idx_t node_idx, std::uint32_t value) {
  if (!per_item_eps_mode_) return;
  if (!wide_per_item_eps_mode_ && value > EPS16_MAX) promote_item_eps_to_wide();
  if (wide_per_item_eps_mode_) wide_per_item_eps_[node_idx] = value;
  else per_item_eps16_[node_idx] = static_cast<std::uint16_t>(value);
}

void SpaceSaving::promote_item_eps_to_wide() {
  if (wide_per_item_eps_mode_) return;
  wide_per_item_eps_.resize(per_item_eps16_.size(), 0);
  for (std::size_t i = 0; i < per_item_eps16_.size(); ++i) {
    wide_per_item_eps_[i] = static_cast<std::uint32_t>(per_item_eps16_[i]);
  }
  wide_per_item_eps_mode_ = true;
}

void SpaceSaving::attach_child(idx_t bucket_idx, idx_t node_idx) {
  auto& b = buckets_[bucket_idx];
  auto& n = nodes_[node_idx];
  n.parent = bucket_idx;
  n.prev = b.tail;
  n.next = NPOS;
  if (b.tail != NPOS) nodes_[b.tail].next = node_idx;
  else b.head = node_idx;
  b.tail = node_idx;
}

void SpaceSaving::detach_child(idx_t bucket_idx, idx_t node_idx) {
  auto& b = buckets_[bucket_idx];
  auto& n = nodes_[node_idx];
  if (n.prev != NPOS) nodes_[n.prev].next = n.next;
  else b.head = n.next;
  if (n.next != NPOS) nodes_[n.next].prev = n.prev;
  else b.tail = n.prev;
  n.prev = NPOS;
  n.next = NPOS;
  n.parent = NPOS;
}

SpaceSaving::idx_t SpaceSaving::make_bucket_after(idx_t where, std::uint32_t value) {
  idx_t idx = NPOS;
  if (!free_buckets_.empty()) {
    idx = free_buckets_.back();
    free_buckets_.pop_back();
    buckets_[idx] = Bucket{};
  } else {
    idx = static_cast<idx_t>(buckets_.size());
    buckets_.push_back(Bucket{});
    if (wide_bucket_values_mode_) wide_bucket_vals_.resize(buckets_.size(), 0);
  }

  auto& nb = buckets_[idx];
  if (!wide_bucket_values_mode_ && value > BUCKET16_MAX) promote_bucket_values_to_wide();
  set_bucket_value(idx, value);

  if (where == NPOS) {
    b_head_ = b_tail_ = idx;
    return idx;
  }

  auto& w = buckets_[where];
  nb.prev = where;
  nb.next = w.next;
  if (w.next != NPOS) buckets_[w.next].prev = idx;
  else b_tail_ = idx;
  w.next = idx;
  return idx;
}

void SpaceSaving::maybe_delete_bucket(idx_t bucket_idx) {
  if (bucket_idx == NPOS) return;
  auto& b = buckets_[bucket_idx];
  if (b.head != NPOS) return;

  if (b.prev != NPOS) buckets_[b.prev].next = b.next;
  else b_head_ = b.next;
  if (b.next != NPOS) buckets_[b.next].prev = b.prev;
  else b_tail_ = b.prev;

  b = Bucket{};
  free_buckets_.push_back(bucket_idx);
}

void SpaceSaving::increment(idx_t node_idx) {
  auto& n = nodes_[node_idx];
  const idx_t cur = n.parent;
  const std::uint32_t cur_cnt = node_count(node_idx);
  if (!wide_counts_ && cur_cnt >= CNT16_MAX) promote_counts_to_wide();
  const std::uint32_t new_val = cur_cnt + 1;
  idx_t nxt = buckets_[cur].next;

  detach_child(cur, node_idx);
  if (!(nxt != NPOS && bucket_value(nxt) == new_val)) {
    nxt = make_bucket_after(cur, new_val);
  }
  attach_child(nxt, node_idx);
  set_node_count(node_idx, new_val);

  maybe_delete_bucket(cur);
}

SpaceSaving::idx_t SpaceSaving::admit_and_increment(const Id128& id) {
  const idx_t bmin_idx = min_bucket();
  assert(bmin_idx != NPOS && "there must be at least one bucket");
  auto& bmin = buckets_[bmin_idx];
  const idx_t victim_idx = bmin.head;
  assert(victim_idx != NPOS && "min bucket must have a node");

  const Id128 old_id = nodes_[victim_idx].id;
  if (node_count(victim_idx) > 0) {
    loc_erase(old_id);
  }

  nodes_[victim_idx].id = id;
  const std::uint32_t bmin_val = bucket_value(bmin_idx);
  if (bmin_val > eps_max_) eps_max_ = bmin_val;
  if (per_item_eps_mode_) set_item_eps(victim_idx, bmin_val);

  if (on_admit_) on_admit_(id);

  increment(victim_idx);
  return victim_idx;
}

void SpaceSaving::destroy_buckets() {
  buckets_.clear();
  free_buckets_.clear();
  b_head_ = NPOS;
  b_tail_ = NPOS;
}

void SpaceSaving::init_zero_bucket() {
  destroy_buckets();
  const idx_t b0 = make_bucket_after(NPOS, 0);

  nodes_.resize(m_);
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    auto& n = nodes_[i];
    n.cnt = 0;
    n.prev = NPOS;
    n.next = NPOS;
    n.parent = NPOS;
    attach_child(b0, static_cast<idx_t>(i));
  }
}

SpaceSaving::SpaceSaving(std::size_t m, bool per_item_eps)
  : m_(m), per_item_eps_mode_(per_item_eps), eps_max_(0) {
  assert(m_ > 0);
  if (m_ >= static_cast<std::size_t>(LOC_NODE_NPOS)) {
    throw std::invalid_argument("SpaceSaving capacity must be <= 65534 for compact LocEntry index");
  }
  if (per_item_eps_mode_) per_item_eps16_.assign(m_, 0);
  loc_reset(m_);
  init_zero_bucket();
}

SpaceSaving::~SpaceSaving() {
  destroy_buckets();
}

void SpaceSaving::reconfigure(std::size_t new_m) {
  if (new_m == 0) new_m = 1;
  if (new_m >= static_cast<std::size_t>(LOC_NODE_NPOS)) {
    throw std::invalid_argument("SpaceSaving capacity must be <= 65534 for compact LocEntry index");
  }
  if (new_m == m_) return;

  m_ = new_m;
  nodes_.assign(m_, {});
  loc_reset(m_);
  wide_counts_ = false;
  wide_cnts_.clear();
  wide_bucket_values_mode_ = false;
  wide_bucket_vals_.clear();
  wide_per_item_eps_mode_ = false;
  wide_per_item_eps_.clear();
  if (per_item_eps_mode_) per_item_eps16_.assign(m_, 0);
  else per_item_eps16_.clear();
  eps_max_ = 0;
  N_ = 0;
  init_zero_bucket();
}

void SpaceSaving::update(const Id128& id, int weight) {
  if (weight <= 0) return;

  for (int t = 0; t < weight; ++t) {
    ++N_;
    const idx_t idx = loc_find(id);
    if (idx != NPOS) {
      increment(idx);
    } else {
      const idx_t admitted = admit_and_increment(id);
      loc_insert(id, admitted);
    }
  }
}

void SpaceSaving::reset_window() {
  destroy_buckets();
  loc_clear();
  wide_counts_ = false;
  wide_cnts_.clear();
  wide_bucket_values_mode_ = false;
  wide_bucket_vals_.clear();
  wide_per_item_eps_mode_ = false;
  wide_per_item_eps_.clear();
  if (per_item_eps_mode_) per_item_eps16_.assign(m_, 0);
  else per_item_eps16_.clear();
  eps_max_ = 0;
  N_ = 0;
  nodes_.assign(m_, {});
  init_zero_bucket();
}

std::size_t SpaceSaving::memory_bytes() const {
  std::size_t bytes = sizeof(*this);
  bytes += nodes_.capacity() * sizeof(Node);
  bytes += buckets_.capacity() * sizeof(Bucket);
  bytes += free_buckets_.capacity() * sizeof(idx_t);
  bytes += loc_slots_.capacity() * sizeof(LocEntry);
  bytes += wide_cnts_.capacity() * sizeof(std::uint32_t);
  bytes += wide_bucket_vals_.capacity() * sizeof(std::uint32_t);
  bytes += per_item_eps16_.capacity() * sizeof(std::uint16_t);
  bytes += wide_per_item_eps_.capacity() * sizeof(std::uint32_t);
  return bytes;
}

Snapshot SpaceSaving::snapshot() const {
  Snapshot s;
  s.N_local = N_;
  s.candidates.reserve(m_);

  for (idx_t b = max_bucket(); b != NPOS; b = buckets_[b].prev) {
    for (idx_t n = buckets_[b].tail; n != NPOS; n = nodes_[n].prev) {
      const std::uint32_t cnt = node_count(n);
      if (cnt) s.candidates.push_back(Cand{nodes_[n].id, cnt});
    }
  }
  return s;
}

SnapshotEx SpaceSaving::snapshot_ex() const {
  SnapshotEx x;
  x.N_local = N_;
  x.q_local = m_;
  x.head_mass = 0;
  x.head_size = 0;
  x.candidates.reserve(m_);
  x.errors.reserve(m_);

  for (idx_t b = max_bucket(); b != NPOS; b = buckets_[b].prev) {
    for (idx_t n = buckets_[b].tail; n != NPOS; n = nodes_[n].prev) {
      const auto& node = nodes_[n];
      const std::uint32_t cnt = node_count(n);
      if (!cnt) continue;
      const std::uint32_t eps = item_eps(n);
      const std::uint32_t lb = (cnt >= eps) ? (cnt - eps) : 0;
      x.candidates.push_back(Cand{node.id, cnt});
      x.errors.push_back(CandErr{lb, eps});
    }
  }
  return x;
}

std::vector<CandWithErr> SpaceSaving::snapshot_with_error() const {
  std::vector<CandWithErr> out;
  out.reserve(m_);
  for (idx_t b = max_bucket(); b != NPOS; b = buckets_[b].prev) {
    for (idx_t n = buckets_[b].tail; n != NPOS; n = nodes_[n].prev) {
      const auto& node = nodes_[n];
      const std::uint32_t cnt = node_count(n);
      if (!cnt) continue;
      const std::uint32_t eps = item_eps(n);
      out.push_back(CandWithErr{node.id, cnt, eps});
    }
  }
  return out;
}

FrequentResult SpaceSaving::query_frequent(double phi) const {
  FrequentResult out;
  if (phi <= 0.0) { out.guaranteed = true; return out; }
  if (phi > 1.0) phi = 1.0;

  const std::uint64_t thresh = static_cast<std::uint64_t>(std::ceil(phi * static_cast<double>(N_)));
  std::uint32_t min_guaranteed = UINT32_MAX;

  for (idx_t b = max_bucket(); b != NPOS; b = buckets_[b].prev) {
    if (bucket_value(b) < thresh) break;
    for (idx_t n = buckets_[b].tail; n != NPOS; n = nodes_[n].prev) {
      const auto& node = nodes_[n];
      const std::uint32_t cnt = node_count(n);
      if (!cnt) continue;
      out.items.push_back(Cand{node.id, cnt});
      const std::uint32_t eps = item_eps(n);
      const std::uint32_t guar = (cnt >= eps) ? (cnt - eps) : 0;
      if (guar < min_guaranteed) min_guaranteed = guar;
    }
  }

  out.guaranteed = (out.items.empty() ? true : (min_guaranteed >= thresh));
  return out;
}

TopKResult SpaceSaving::query_topk(std::size_t k) const {
  TopKResult out;
  if (k == 0) { out.guaranteed = true; out.order = true; return out; }

  std::vector<idx_t> vec;
  vec.reserve(m_);
  for (idx_t b = max_bucket(); b != NPOS; b = buckets_[b].prev) {
    for (idx_t n = buckets_[b].tail; n != NPOS; n = nodes_[n].prev) {
      if (node_count(n)) vec.push_back(n);
    }
  }

  const std::size_t kk = std::min(k, vec.size());
  out.items.reserve(kk);
  for (std::size_t i = 0; i < kk; ++i) {
    const auto& n = nodes_[vec[i]];
    out.items.push_back(Cand{n.id, node_count(vec[i])});
  }

  const std::uint32_t ck1 = (vec.size() > k) ? node_count(vec[k]) : 0;
  bool guaranteed = true;
  bool order_ok = true;

  for (std::size_t i = 0; i < kk; ++i) {
    const std::uint32_t cnt = node_count(vec[i]);
    const std::uint32_t eps = item_eps(vec[i]);
    const std::uint32_t guar = (cnt >= eps) ? (cnt - eps) : 0;
    if (guar < ck1) guaranteed = false;
    if (i + 1 < vec.size()) {
      const std::uint32_t next_cnt = node_count(vec[i + 1]);
      if (guar < next_cnt) order_ok = false;
    }
  }

  out.guaranteed = guaranteed;
  out.order = order_ok;
  return out;
}

} // namespace hh
