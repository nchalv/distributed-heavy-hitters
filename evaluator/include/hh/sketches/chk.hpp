#pragma once
#include "hh/sketches/isketch.hpp"
#include "hh/core/id128.hpp"

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace hh {

/**
 * CHK (Cuckoo HeavyKeeper, adapted) as an ISketch:
 * - Two cuckoo tables; each bucket has 3 entries:
 *     entries[0] = lobby; entries[1..2] = heavy candidates.
 * - Keys are handled via a 16-bit fingerprint for placement, but we also
 *   store the last-seen Id128 in the entry so we can emit snapshots by Id128.
 * - Promotions from lobby → heavy use a probability proportional to counters.
 * - Lobby decays follow HeavyKeeper logic with base `decay_base_`.
 *
 * Construction:
 *   CHK(B, L=16, decay=1.08, theta_phi=0.01)
 *   B is rounded up to power-of-two internally (required by xor-indexing).
 */
class CHK : public ISketch {
public:
  using fingerprint_t = std::uint16_t;
  using counter_t     = std::uint32_t;

  CHK(std::size_t buckets,
      std::size_t L = 16,
      double decay = 1.08,
      double theta_phi = 0.01);

  // ISketch API
  void update(const Id128& id, int weight) override;
  Snapshot snapshot() const override;
  SnapshotEx snapshot_ex() const override;
  void reset_window() override;
  void set_admission_callback(AdmitFn cb) override { on_admit_ = std::move(cb); }

private:
  struct Entry {
    fingerprint_t fp{0};
    counter_t     cnt{0};
    Id128         last_id{};     // last key that hit this entry (for reporting)
    bool empty() const { return cnt == 0; }
  };

  struct Bucket {
    static constexpr std::size_t ENTRIES_PER_BUCKET = 3; // [0]=lobby, [1..2]=heavy
    Entry e[ENTRIES_PER_BUCKET];

    Entry&       lobby()             { return e[0]; }
    const Entry& lobby() const       { return e[0]; }

    Entry& smallest_heavy() {
      // among e[1], e[2]
      return (e[1].cnt <= e[2].cnt) ? e[1] : e[2];
    }
  };

  // params
  std::size_t buckets_;           // power-of-two
  std::size_t L_;                 // promotion threshold
  double      decay_base_;        // HeavyKeeper decay base (e.g., 1.08)
  double      theta_phi_;         // θ = 1/n (used in "is heavy" kick-chain guard)

  // state
  std::array<std::vector<Bucket>, 2> tables_; // two tables of equal size
  std::uint64_t N_local_{0};

  // RNG for probabilistic decay / promotion
  std::mt19937_64 rng_;
  std::uniform_real_distribution<double> U_{0.0, 1.0};

  // precomputed expectations up to MAX_COUNTER (small bounds suffice for lobby behavior)
  static constexpr std::size_t MAX_COUNTER = 16;
  std::array<double, MAX_COUNTER + 1> decay_expectations_{};
  std::array<double, MAX_COUNTER + 1> min_decay_amounts_{};

  AdmitFn on_admit_{};

  // helpers
  static inline bool is_pow2(std::size_t x) { return x && !(x & (x - 1)); }
  static inline std::size_t round_up_pow2(std::size_t x) {
    if (is_pow2(x)) return x;
    if (x == 0) return 1;
    --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
#if INTPTR_MAX == INT64_MAX
    x |= x >> 32;
#endif
    return x + 1;
  }

  static fingerprint_t fp16_from_id(const Id128& id);
  static std::size_t   idx1_from_id(const Id128& id, std::size_t buckets);
  static std::size_t   alt_index(fingerprint_t fp, std::size_t idx, std::size_t buckets);

  void init_decay_tables();

  // update building blocks
  bool update_heavy(fingerprint_t fp, const Id128& id, std::size_t idx1, std::size_t idx2, int w);
  bool update_lobby_hit(fingerprint_t fp, const Id128& id, std::size_t idx1, std::size_t idx2, int w);
  bool place_if_empty_lobby(fingerprint_t fp, const Id128& id, std::size_t idx1, std::size_t idx2, int w);

  counter_t decay_counter(counter_t current, int weight);

  bool try_promote_and_kick(Entry& lobby, Entry& smallest, std::size_t t, std::size_t idx);
  void kick_chain(Entry kicked, std::size_t t, std::size_t idx);

  bool is_heavy(counter_t c) const {
    // relaxed guard; avoids unbounded cuckoo walks for non-heavy items
    constexpr double HEAVY_RATIO = 0.8;
    return static_cast<double>(c) >= static_cast<double>(N_local_) * (theta_phi_ * HEAVY_RATIO);
  }
};

} // namespace hh
