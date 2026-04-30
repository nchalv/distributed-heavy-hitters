#include "hh/coord/coordinator.hpp"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cmath>

namespace hh {

std::string Coordinator::id128_hex(const Id128& id) {
  static const char* HEX = "0123456789abcdef";
  std::string s; s.resize(32);
  for (int i = 0; i < 16; ++i) {
    const unsigned v = id.b[i];
    s[2 * i + 0] = HEX[(v >> 4) & 0xF];
    s[2 * i + 1] = HEX[v & 0xF];
  }
  return s;
}

GlobalResultLB Coordinator::reduce_global_with_lb(
    const std::vector<SnapshotEx>& snaps_ex,
    const std::vector<const ArenaMap*>& maps,
    std::size_t n_param,
    ReduceTelemetry* telemetry)
{
  GlobalResultLB R{};
  if (n_param == 0) return R;
  if (telemetry) *telemetry = ReduceTelemetry{};

  // 1) N_global and strict HH threshold:
  //    p(k) > 1/n  <=>  f(k) >= floor(N/n) + 1.
  for (const auto& s : snaps_ex) R.N_global += s.N_local;
  R.threshold = (R.N_global / n_param) + 1;

  // 2) Aggregate per id
  struct Agg {
    std::uint64_t est=0, lb=0, reporters=0;
    double mass_reporters=0.0;
    std::uint64_t eps_sum=0;      // sum of per-item eps over reporters (SS only)
    bool has_ss_reporter{false};  // true if reported by at least one SS partition
  };
  std::unordered_map<Id128, Agg, Id128Hash> agg;
  agg.reserve(1024);

  // Keep per-partition mass, present-id sets, and SS min-counter terms c_i.
  std::vector<double> part_mass; part_mass.reserve(snaps_ex.size());
  std::vector<std::unordered_set<Id128, Id128Hash>> present_by_part;
  present_by_part.reserve(snaps_ex.size());
  std::vector<double> min_counter_by_part; min_counter_by_part.reserve(snaps_ex.size());
  for (const auto& s : snaps_ex) part_mass.push_back(static_cast<double>(s.N_local));

  for (std::size_t idx=0; idx<snaps_ex.size(); ++idx) {
    const auto& s = snaps_ex[idx];
    // dedup ids per partition for coverage and reporter tests
    std::unordered_set<Id128, Id128Hash> pres;
    pres.reserve(s.candidates.size()*2);
    for (const auto& c : s.candidates) pres.insert(c.id);
    present_by_part.push_back(std::move(pres));

    // Space-Saving min-counter proxy c_i:
    // - if summary not full (|S_i| < q_i): c_i = 0
    // - else c_i = min retained counter
    double c_i = 0.0;
    if (s.q_local > 0 && s.candidates.size() >= s.q_local) {
      std::uint32_t mn = UINT32_MAX;
      for (const auto& c : s.candidates) mn = std::min(mn, c.est);
      c_i = (mn == UINT32_MAX) ? 0.0 : static_cast<double>(mn);
    }
    min_counter_by_part.push_back(c_i);

    const bool has_bounds = s.has_error_bounds();
    for (std::size_t ci = 0; ci < s.candidates.size(); ++ci) {
      const auto& c = s.candidates[ci];
      auto& a = agg[c.id];
      a.est += c.est;
      if (has_bounds) {
        a.lb  += s.errors[ci].lb;
        a.eps_sum += s.errors[ci].eps;
      }
    }
    for (const auto& id : present_by_part.back()) {
      auto& a = agg[id];
      a.reporters += 1;
      a.mass_reporters += part_mass[idx];
      if (s.q_local > 0) a.has_ss_reporter = true;
    }
  }

  if (telemetry) {
    std::size_t agg_bytes = 0;
    agg_bytes += sizeof(agg);
    agg_bytes += agg.bucket_count() * sizeof(void*);
    agg_bytes += agg.size() * sizeof(decltype(agg)::value_type);

    std::size_t presence_bytes = 0;
    presence_bytes += sizeof(part_mass) + part_mass.capacity() * sizeof(decltype(part_mass)::value_type);
    presence_bytes += sizeof(present_by_part)
                   + present_by_part.capacity() * sizeof(decltype(present_by_part)::value_type);
    presence_bytes += sizeof(min_counter_by_part)
                   + min_counter_by_part.capacity() * sizeof(decltype(min_counter_by_part)::value_type);
    for (const auto& pres : present_by_part) {
      presence_bytes += pres.bucket_count() * sizeof(void*);
      presence_bytes += pres.size() * sizeof(Id128);
    }

    telemetry->agg_bytes = agg_bytes;
    telemetry->presence_bytes = presence_bytes;
    telemetry->total_peak_bytes = std::max(telemetry->total_peak_bytes, agg_bytes + presence_bytes);
  }

  // 3) Build results and resolve keys
  std::vector<GlobalItemLB> items; items.reserve(agg.size());
  for (const auto& kv : agg) {
    GlobalItemLB gi;
    gi.id  = kv.first;
    gi.est = kv.second.est;
    gi.lb  = kv.second.lb;
    gi.reporters = kv.second.reporters;
    gi.omega = (R.N_global > 0) ? (kv.second.mass_reporters / static_cast<double>(R.N_global)) : 0.0;

    // Routing-agnostic SS certification envelope:
    //   f_hat - sum_{reporters} eps_i <= f <= f_hat + sum_{non-reporters} c_j
    if (kv.second.has_ss_reporter) {
      const double inflation = static_cast<double>(kv.second.eps_sum);
      double hidden_non_reporter_mass = 0.0;
      for (std::size_t i = 0; i < snaps_ex.size(); ++i) {
        if (present_by_part[i].find(kv.first) == present_by_part[i].end()) {
          hidden_non_reporter_mass += min_counter_by_part[i];
        }
      }

      const double cert_lb_d = static_cast<double>(gi.est) - inflation;
      gi.cert_lb = cert_lb_d > 0.0 ? static_cast<std::uint64_t>(cert_lb_d) : 0ull;
      const double cert_ub_d = static_cast<double>(gi.est) + hidden_non_reporter_mass;
      gi.cert_ub = static_cast<std::uint64_t>(std::ceil(std::max(0.0, cert_ub_d)));
      gi.guaranteed = (gi.cert_lb >= R.threshold);
    } else {
      // Fallback for non-SS sketches (no per-item epsilon / c_i telemetry contract).
      gi.cert_lb = gi.lb;
      gi.cert_ub = gi.est;
      gi.guaranteed = (gi.lb >= R.threshold);
    }

    std::string bytes;
    for (const auto* amap : maps) {
      if (amap && amap->lookup(kv.first, bytes)) { gi.key = bytes; break; }
    }
    items.push_back(std::move(gi));
  }

  if (telemetry) {
    std::size_t items_bytes = 0;
    items_bytes += sizeof(items);
    items_bytes += items.capacity() * sizeof(decltype(items)::value_type);
    for (const auto& gi : items) items_bytes += gi.key.capacity();
    telemetry->items_bytes = items_bytes;
    telemetry->total_peak_bytes = std::max(
        telemetry->total_peak_bytes,
        telemetry->agg_bytes + telemetry->presence_bytes + telemetry->items_bytes);
  }

  std::sort(items.begin(), items.end(), [](const auto& a, const auto& b){
    if (a.est != b.est) return a.est > b.est;
    return a.key < b.key;
  });

  R.items = std::move(items);
  return R;
}

GlobalResultLB Coordinator::reduce_hl_bucketwise(
    const std::vector<HLBucketSnapshot>& snaps_hl,
    const std::vector<const ArenaMap*>& maps,
    std::size_t n_param,
    ReduceTelemetry* telemetry)
{
  GlobalResultLB R{};
  if (n_param == 0) return R;
  if (telemetry) *telemetry = ReduceTelemetry{};
  if (snaps_hl.empty()) return R;

  for (const auto& s : snaps_hl) R.N_global += s.N_local;
  R.threshold = (R.N_global / n_param) + 1;

  std::size_t w = 0;
  std::size_t d = 0;
  for (const auto& s : snaps_hl) {
    if (s.w == 0 || s.d == 0) continue;
    if (w == 0) w = s.w;
    if (d == 0) d = s.d;
    w = std::min(w, s.w);
    d = std::min(d, s.d);
  }
  if (w == 0 || d == 0) return R;

  std::vector<std::unordered_map<Id128, std::uint64_t, Id128Hash>> by_bucket;
  by_bucket.resize(w);
  for (const auto& s : snaps_hl) {
    for (const auto& c : s.candidates) {
      if (c.bucket >= w) continue;
      by_bucket[c.bucket][c.id] += c.est;
    }
  }

  std::unordered_map<Id128, std::uint64_t, Id128Hash> agg;
  agg.reserve(w * d);
  std::size_t scratch_peak_bytes = 0;
  for (std::size_t bi = 0; bi < w; ++bi) {
    auto& mp = by_bucket[bi];
    if (mp.empty()) continue;

    std::vector<std::pair<Id128, std::uint64_t>> vec;
    vec.reserve(mp.size());
    for (const auto& kv : mp) vec.push_back(kv);
    scratch_peak_bytes = std::max(
        scratch_peak_bytes,
        sizeof(vec) + vec.capacity() * sizeof(decltype(vec)::value_type));

    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b){
      if (a.second != b.second) return a.second > b.second;
      return Id128Hash{}(a.first) < Id128Hash{}(b.first);
    });

    const std::size_t keep = std::min<std::size_t>(d, vec.size());
    for (std::size_t i = 0; i < keep; ++i) {
      agg[vec[i].first] += vec[i].second;
    }
  }

  std::vector<GlobalItemLB> items;
  items.reserve(agg.size());
  for (const auto& kv : agg) {
    GlobalItemLB gi;
    gi.id = kv.first;
    gi.est = kv.second;
    gi.lb = 0;
    gi.reporters = 0;
    gi.omega = 0.0;
    gi.cert_lb = 0;
    gi.cert_ub = gi.est;
    gi.guaranteed = false;

    std::string bytes;
    for (const auto* amap : maps) {
      if (amap && amap->lookup(kv.first, bytes)) { gi.key = bytes; break; }
    }
    items.push_back(std::move(gi));
  }

  if (telemetry) {
    std::size_t by_bucket_bytes = 0;
    by_bucket_bytes += sizeof(by_bucket);
    by_bucket_bytes += by_bucket.capacity() * sizeof(decltype(by_bucket)::value_type);
    for (const auto& mp : by_bucket) {
      by_bucket_bytes += sizeof(mp);
      by_bucket_bytes += mp.bucket_count() * sizeof(void*);
      by_bucket_bytes += mp.size() * sizeof(std::unordered_map<Id128, std::uint64_t, Id128Hash>::value_type);
    }

    std::size_t agg_bytes = 0;
    agg_bytes += sizeof(agg);
    agg_bytes += agg.bucket_count() * sizeof(void*);
    agg_bytes += agg.size() * sizeof(decltype(agg)::value_type);

    telemetry->agg_bytes = by_bucket_bytes + agg_bytes;
    telemetry->presence_bytes = scratch_peak_bytes;

    std::size_t items_bytes = 0;
    items_bytes += sizeof(items);
    items_bytes += items.capacity() * sizeof(decltype(items)::value_type);
    for (const auto& gi : items) items_bytes += gi.key.capacity();
    telemetry->items_bytes = items_bytes;
    telemetry->total_peak_bytes = std::max(
        telemetry->total_peak_bytes,
        telemetry->agg_bytes + telemetry->presence_bytes + telemetry->items_bytes);
  }

  std::sort(items.begin(), items.end(), [](const auto& a, const auto& b){
    if (a.est != b.est) return a.est > b.est;
    return a.key < b.key;
  });

  R.items = std::move(items);
  return R;
}

} // namespace hh
