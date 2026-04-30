#pragma once
#include "hh/coord/coordinator.hpp"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace hh {

struct WindowMetrics {
  // HH precision/recall are computed at strict threshold floor(N_global / n_param) + 1 using oracle HH labels.
  double hh_precision{0.0};
  double hh_recall{0.0};
  double hh_f1{0.0};
  std::uint64_t hh_tp{0};
  std::uint64_t hh_fp{0};
  std::uint64_t hh_fn{0};
  std::size_t hh_items{0};
  double abs_err_sum{0.0};
  double rel_err_sum{0.0};
  double aae{0.0};                // average absolute error over oracle HH items
  double are{0.0};                // average relative error over oracle HH items
  std::optional<double> topk_overlap{}; // present only if requested
  std::size_t k_used{0};
  std::uint64_t N_global{0};
};

struct AggregateMetrics {
  double hh_precision_avg{0.0};
  double hh_recall_avg{0.0};
  double hh_f1_avg{0.0};
  double aae_avg{0.0};
  double are_avg{0.0};
  std::optional<double> topk_overlap_avg{};
  std::size_t windows{0};
};

WindowMetrics eval_vs_oracle(const GlobalResultLB& oracle,
                             const GlobalResultLB& method,
                             std::optional<std::size_t> k_opt);  // <-- optional

AggregateMetrics mean_of(const std::vector<WindowMetrics>& ws);

} // namespace hh
