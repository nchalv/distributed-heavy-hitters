#pragma once
#include "hh/sketches/isketch.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hh {

struct CandWithErr {
  Id128 id;
  std::uint32_t est;
  std::uint32_t eps;
};

struct FrequentResult {
  std::vector<Cand> items;
  bool guaranteed = false;
};

struct TopKResult {
  std::vector<Cand> items;
  bool guaranteed = false;
  bool order      = false;
};

class SpaceSaving : public ISketch {
public:
  explicit SpaceSaving(std::size_t m, bool per_item_eps = true);
  ~SpaceSaving() override;

  void update(const Id128& id, int weight=1) override;
  Snapshot snapshot() const override;
  void reset_window() override;
  std::size_t memory_bytes() const override;

  SnapshotEx snapshot_ex() const override;
  void set_admission_callback(ISketch::AdmitFn cb) override { on_admit_ = std::move(cb); }

  std::vector<CandWithErr> snapshot_with_error() const;
  FrequentResult query_frequent(double phi) const;
  TopKResult     query_topk(std::size_t k) const;
  void reconfigure(std::size_t new_m) override;
  std::size_t capacity() const { return m_; }

private:
  using idx_t = std::uint16_t;
  static constexpr idx_t NPOS = UINT16_MAX;

  struct Bucket;
  struct Node {
    Id128 id{};
    std::uint16_t cnt{0}; // compact mode; auto-promoted via wide_counts_ on overflow risk
    idx_t prev{NPOS};
    idx_t next{NPOS};
    idx_t parent{NPOS};
  };
  struct Bucket {
    std::uint16_t value{0}; // compact mode; auto-promoted via wide_bucket_values_ on overflow risk
    idx_t prev{NPOS};
    idx_t next{NPOS};
    idx_t head{NPOS};
    idx_t tail{NPOS};
  };

  struct LocEntry {
    // Packed to 4 bytes:
    // - node: 16 bits (valid when state==filled)
    // - meta low 2 bits: state (0=empty, 1=filled, 2=tombstone)
    // - meta high 14 bits: short hash tag
    std::uint16_t node{0};
    std::uint16_t meta{0};
  };

  void increment(idx_t node_idx);
  void attach_child(idx_t bucket_idx, idx_t node_idx);
  void detach_child(idx_t bucket_idx, idx_t node_idx);
  idx_t make_bucket_after(idx_t where, std::uint32_t value);
  void maybe_delete_bucket(idx_t bucket_idx);
  idx_t admit_and_increment(const Id128& id);
  idx_t min_bucket() const { return b_head_; }
  idx_t max_bucket() const { return b_tail_; }
  void destroy_buckets();
  void init_zero_bucket();
  std::uint32_t node_count(idx_t node_idx) const;
  void set_node_count(idx_t node_idx, std::uint32_t value);
  void promote_counts_to_wide();
  std::uint32_t bucket_value(idx_t bucket_idx) const;
  void set_bucket_value(idx_t bucket_idx, std::uint32_t value);
  void promote_bucket_values_to_wide();
  std::uint32_t item_eps(idx_t node_idx) const;
  void set_item_eps(idx_t node_idx, std::uint32_t value);
  void promote_item_eps_to_wide();

  // Flat hash index: Id128 -> node index
  static std::size_t next_pow2(std::size_t x);
  static std::size_t loc_hash(const Id128& id);
  static std::uint16_t loc_tag(std::size_t h);
  void loc_reset(std::size_t expected_size);
  void loc_clear();
  idx_t loc_find(const Id128& id) const;
  void loc_insert(const Id128& id, idx_t node_idx);
  void loc_erase(const Id128& id);
  void loc_rehash(std::size_t new_capacity);

  std::size_t m_;
  std::vector<Node> nodes_;
  std::vector<Bucket> buckets_;
  std::vector<idx_t> free_buckets_;
  idx_t b_head_{NPOS};
  idx_t b_tail_{NPOS};
  std::vector<LocEntry> loc_slots_;
  std::size_t loc_size_{0};
  std::size_t loc_tombs_{0};
  bool wide_counts_{false};
  std::vector<std::uint32_t> wide_cnts_;
  bool wide_bucket_values_mode_{false};
  std::vector<std::uint32_t> wide_bucket_vals_;
  bool per_item_eps_mode_{false};
  bool wide_per_item_eps_mode_{false};
  std::vector<std::uint16_t> per_item_eps16_;
  std::vector<std::uint32_t> wide_per_item_eps_;
  std::uint32_t eps_max_{0}; // single per-sketch max admission error (local overestimation bound)
  std::uint64_t N_{0};
  ISketch::AdmitFn on_admit_{};
};

} // namespace hh
