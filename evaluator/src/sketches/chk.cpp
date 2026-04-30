#include "hh/sketches/chk.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace hh {

// ---------- small hash/fp utilities (id128 → fp, indices) ----------
CHK::fingerprint_t CHK::fp16_from_id(const Id128& id) {
  std::uint64_t lo = 0, hi = 0;
  std::memcpy(&lo, id.b.data() + 0, 8);
  std::memcpy(&hi, id.b.data() + 8, 8);
  std::uint64_t x = lo ^ (hi + 0x9e3779b185ebca87ull + (lo << 6) + (lo >> 2));
  x ^= (x >> 33);
  x *= 0xff51afd7ed558ccdULL;
  x ^= (x >> 33);
  return static_cast<fingerprint_t>(x & 0xFFFFu);
}

std::size_t CHK::idx1_from_id(const Id128& id, std::size_t buckets) {
  std::uint64_t lo = 0, hi = 0;
  std::memcpy(&lo, id.b.data() + 0, 8);
  std::memcpy(&hi, id.b.data() + 8, 8);
  std::uint64_t x = lo ^ (hi + 0x9e3779b185ebca87ull + (lo << 6) + (lo >> 2));
  x ^= (x >> 29);
  x *= 0x9ddfea08eb382d69ULL;
  x ^= (x >> 32);
  // power-of-two mask
  return static_cast<std::size_t>((x >> 32) & (buckets - 1));
}

std::size_t CHK::alt_index(fingerprint_t fp, std::size_t idx, std::size_t buckets) {
  // standard XOR-based alt index (cuckoo)
  return (idx ^ (0x5bd1e995u * static_cast<std::uint32_t>(fp))) & (buckets - 1);
}

// ---------- ctor / reset ----------
CHK::CHK(std::size_t buckets, std::size_t L, double decay, double theta_phi)
  : buckets_(round_up_pow2(std::max<std::size_t>(1, buckets))),
    L_(L ? L : 1),
    decay_base_(decay > 1.0 ? decay : 1.08),
    theta_phi_(theta_phi > 0.0 ? theta_phi : 0.01),
    rng_(std::random_device{}())
{
  tables_[0].resize(buckets_);
  tables_[1].resize(buckets_);
  init_decay_tables();
}

void CHK::reset_window() {
  N_local_ = 0;
  for (auto t = 0; t < 2; ++t) {
    for (auto& b : tables_[t]) {
      for (auto& e : b.e) { e.cnt = 0; e.fp = 0; e.last_id = {}; }
    }
  }
}

// ---------- decay expectations init ----------
void CHK::init_decay_tables() {
  decay_expectations_.fill(0.0);
  min_decay_amounts_.fill(0.0);
  // E[c] = sum_{i=0}^{c-1} (decay_base)^i, with E[0]=0
  for (int c = 1; c <= static_cast<int>(MAX_COUNTER); ++c) {
    decay_expectations_[c] = decay_expectations_[c - 1] + std::pow(decay_base_, c - 1);
    min_decay_amounts_[c]  = decay_expectations_[c] - decay_expectations_[c - 1];
  }
}

// ---------- lobby decay ----------
CHK::counter_t CHK::decay_counter(counter_t current, int weight) {
  if (current == 0) return 0;

  if (weight == 1) {
    const double p = std::pow(decay_base_, -static_cast<double>(current));
    return (U_(rng_) < p) ? static_cast<counter_t>(current - 1) : current;
  }

  if (weight > 1 && current <= MAX_COUNTER && weight < min_decay_amounts_[current]) {
    const double p = static_cast<double>(weight) / min_decay_amounts_[current];
    return (U_(rng_) < p) ? static_cast<counter_t>(current - 1) : current;
  }

  if (current <= MAX_COUNTER && weight >= decay_expectations_[current]) {
    return 0;
  }

  // binary search: find smallest x s.t. E[x] + weight >= E[current]
  int left = 0, right = static_cast<int>(current);
  const double target = (current <= MAX_COUNTER) ? (decay_expectations_[current] - weight)
                                                 : (static_cast<double>(current) - weight); // coarse fallback
  while (left < right) {
    int mid = left + (right - left) / 2;
    const double Emid = (mid <= static_cast<int>(MAX_COUNTER)) ? decay_expectations_[mid]
                                                               : static_cast<double>(mid); // coarse fallback
    if (Emid >= target) right = mid;
    else                left  = mid + 1;
  }
  return static_cast<counter_t>(left);
}

// ---------- promotions / kickouts ----------
bool CHK::try_promote_and_kick(Entry& lobby, Entry& smallest, std::size_t t, std::size_t idx) {
  // empty target? just move
  if (smallest.empty()) {
    std::swap(smallest, lobby);
    return true;
  }

  // probabilistic promotion if target is larger
  if (smallest.cnt > lobby.cnt && smallest.cnt > L_) {
    const double num = static_cast<double>(std::max<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(lobby.cnt) - static_cast<std::ptrdiff_t>(L_)));
    const double den = static_cast<double>(std::max<std::ptrdiff_t>(1, static_cast<std::ptrdiff_t>(smallest.cnt) - static_cast<std::ptrdiff_t>(L_)));
    const double prob = std::min(1.0, num / den);
    if (U_(rng_) >= prob) return false;
  }

  Entry kicked = smallest;
  smallest = lobby;         // promote lobby
  lobby = Entry{};          // clear lobby
  kick_chain(kicked, t, idx);
  return true;
}

void CHK::kick_chain(Entry kicked, std::size_t t, std::size_t idx) {
  static constexpr std::size_t MAX_KICKS = 10;
  std::size_t kicks = 0;

  while (kicks < MAX_KICKS) {
    if (!is_heavy(kicked.cnt)) return; // avoid roaming non-heavy items

    t   = 1 - t;
    idx = alt_index(kicked.fp, idx, buckets_);

    Bucket& b = tables_[t][idx];
    Entry&  s = b.smallest_heavy();

    if (s.empty()) { s = kicked; return; }
    std::swap(s, kicked);
    ++kicks;
  }
}

// ---------- update helpers ----------
bool CHK::update_heavy(fingerprint_t fp, const Id128& id, std::size_t idx1, std::size_t idx2, int w) {
  std::size_t empty_table = static_cast<std::size_t>(-1);
  std::size_t empty_pos   = static_cast<std::size_t>(-1);

  for (std::size_t t = 0; t < 2; ++t) {
    std::size_t idx = (t == 0) ? idx1 : idx2;
    Bucket& b = tables_[t][idx];

    for (std::size_t i = 1; i < Bucket::ENTRIES_PER_BUCKET; ++i) {
      Entry& e = b.e[i];
      if (!e.empty()) {
        if (e.fp == fp) {
          e.cnt += static_cast<counter_t>(w);
          e.last_id = id;
          return true;
        }
      } else if (empty_table == static_cast<std::size_t>(-1)) {
        empty_table = t; empty_pos = i;
      }
    }
  }

  if (empty_table != static_cast<std::size_t>(-1)) {
    std::size_t idx = (empty_table == 0) ? idx1 : idx2;
    Entry& e = tables_[empty_table][idx].e[empty_pos];
    e.fp = fp; e.cnt = static_cast<counter_t>(w); e.last_id = id;
    return true;
  }

  return false;
}

bool CHK::update_lobby_hit(fingerprint_t fp, const Id128& id, std::size_t idx1, std::size_t idx2, int w) {
  for (std::size_t t = 0; t < 2; ++t) {
    std::size_t idx = (t == 0) ? idx1 : idx2;
    Bucket& b = tables_[t][idx];
    Entry&  L = b.lobby();

    if (!L.empty() && L.fp == fp) {
      L.cnt += static_cast<counter_t>(w);
      L.last_id = id;

      if (L.cnt >= L_) {
        Entry& s = b.smallest_heavy();
        // promote if possible
        (void)try_promote_and_kick(L, s, t, idx);
        if (!L.empty() && L.cnt > L_) L.cnt = static_cast<counter_t>(L_);
      }
      return true;
    }
  }
  return false;
}

bool CHK::place_if_empty_lobby(fingerprint_t fp, const Id128& id, std::size_t idx1, std::size_t idx2, int w) {
  for (std::size_t t = 0; t < 2; ++t) {
    std::size_t idx = (t == 0) ? idx1 : idx2;
    Bucket& b = tables_[t][idx];
    Entry&  L = b.lobby();

    if (L.empty()) {
      L.fp = fp; L.cnt = static_cast<counter_t>(w); L.last_id = id;
      if (L.cnt >= L_) {
        Entry& s = b.smallest_heavy();
        (void)try_promote_and_kick(L, s, t, idx);
        if (!L.empty() && L.cnt > L_) L.cnt = static_cast<counter_t>(L_);
      }
      return true;
    }
  }
  return false;
}

// ---------- update ----------
void CHK::update(const Id128& id, int weight) {
  if (weight <= 0) return;
  N_local_ += static_cast<std::uint64_t>(weight);

  const fingerprint_t fp = fp16_from_id(id);
  const std::size_t   i1 = idx1_from_id(id, buckets_);
  const std::size_t   i2 = alt_index(fp, i1, buckets_);

  // 1) heavy hit?
  if (update_heavy(fp, id, i1, i2, weight)) return;

  // 2) lobby hit?
  if (update_lobby_hit(fp, id, i1, i2, weight)) return;

  // 3) empty lobby anywhere?
  if (place_if_empty_lobby(fp, id, i1, i2, weight)) return;

  // 4) choose consistent lobby; decay and maybe replace
  const std::size_t t   = (fp & 1u);
  const std::size_t idx = (t == 0) ? i1 : i2;

  Bucket& b = tables_[t][idx];
  Entry&  L = b.lobby();

  const counter_t old = L.cnt;
  const counter_t nc  = decay_counter(old, weight);

  if (nc == 0) {
    // replace with new fp; approximate compensation for expected decay
    int compensate = 0;
    if (old <= MAX_COUNTER) compensate = static_cast<int>(std::lround(decay_expectations_[old]));
    int new_count = weight - compensate;
    if (new_count < 1) new_count = 1;

    L.fp = fp;
    L.cnt = static_cast<counter_t>(new_count);
    L.last_id = id;
  } else {
    L.cnt = nc;
  }

  if (L.cnt >= L_) {
    Entry& s = b.smallest_heavy();
    (void)try_promote_and_kick(L, s, t, idx);
    if (!L.empty() && L.cnt > L_) L.cnt = static_cast<counter_t>(L_);
  }
}

// ---------- snapshots ----------
Snapshot CHK::snapshot() const {
  Snapshot s{};
  s.N_local = N_local_;

  std::unordered_map<Id128, std::size_t, Id128Hash> agg;
  agg.reserve(buckets_ * 2);

  for (int t = 0; t < 2; ++t) {
    for (const auto& b : tables_[t]) {
      for (std::size_t i = 0; i < Bucket::ENTRIES_PER_BUCKET; ++i) {
        const auto& e = b.e[i];
        if (!e.empty()) agg[e.last_id] += static_cast<std::size_t>(e.cnt);
      }
    }
  }

  s.candidates.reserve(agg.size());
  for (const auto& kv : agg) {
    s.candidates.push_back({kv.first, static_cast<std::uint32_t>(kv.second)});
  }
  return s;
}

SnapshotEx CHK::snapshot_ex() const {
  SnapshotEx sx{};
  sx.N_local = N_local_;
  sx.q_local = 0;
  sx.head_mass = 0;
  sx.head_size = 0;

  std::unordered_map<Id128, std::size_t, Id128Hash> agg;
  agg.reserve(buckets_ * 2);

  for (int t = 0; t < 2; ++t) {
    for (const auto& b : tables_[t]) {
      for (std::size_t i = 0; i < Bucket::ENTRIES_PER_BUCKET; ++i) {
        const auto& e = b.e[i];
        if (!e.empty()) agg[e.last_id] += static_cast<std::size_t>(e.cnt);
      }
    }
  }

  sx.candidates.reserve(agg.size());
  for (const auto& kv : agg) {
    sx.candidates.push_back({kv.first, static_cast<std::uint32_t>(kv.second)});
  }
  return sx;
}

} // namespace hh
