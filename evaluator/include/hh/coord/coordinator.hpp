#pragma once
#include "hh/sketches/isketch.hpp"  // SnapshotEx, Id128
#include "hh/sketches/heavylocker.hpp" // HLBucketSnapshot
#include "hh/core/arena_map.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace hh {

struct GlobalItemLB {
  Id128 id;
  std::string key;
  std::uint64_t est{0};
  std::uint64_t lb{0};
  std::uint64_t reporters{0};  // number of partitions that reported this id
  double omega{0.0};           // coverage: mass fraction of reporting partitions (Eq. 1)
  std::uint64_t cert_lb{0};    // certified lower bound (routing-agnostic SS envelope)
  std::uint64_t cert_ub{0};    // certified upper bound (routing-agnostic SS envelope)
  bool guaranteed{false};
};

struct GlobalResultLB {
  std::vector<GlobalItemLB> items;
  std::uint64_t N_global{0};
  std::uint64_t threshold{0};
};

struct ReduceTelemetry {
  std::size_t agg_bytes{0};
  std::size_t presence_bytes{0};
  std::size_t items_bytes{0};
  std::size_t total_peak_bytes{0};
};

class Coordinator {
public:
  static std::string id128_hex(const Id128& id);

  // Unified LB-aware reducer (works for all sketches; SS provides lb, others 0)
  static GlobalResultLB reduce_global_with_lb(
      const std::vector<SnapshotEx>& snaps_ex,
      const std::vector<const ArenaMap*>& maps,
      std::size_t n_param,
      ReduceTelemetry* telemetry = nullptr);

  // HeavyLocker bucket-wise merge:
  // per-bucket cross-node aggregation -> top-d pruning -> global id aggregation.
  static GlobalResultLB reduce_hl_bucketwise(
      const std::vector<HLBucketSnapshot>& snaps_hl,
      const std::vector<const ArenaMap*>& maps,
      std::size_t n_param,
      ReduceTelemetry* telemetry = nullptr);
};

} // namespace hh
