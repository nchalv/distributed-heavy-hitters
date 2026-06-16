#include "hh/sizing/sizing.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace hh {
namespace sizing {

static double quantile_inplace(std::vector<double>& v, double alpha) {
  if (v.empty()) return 0.0;
  if (alpha <= 0.0) return *std::min_element(v.begin(), v.end());
  if (alpha >= 1.0) return *std::max_element(v.begin(), v.end());
  const std::size_t support = std::max<std::size_t>(
      std::size_t{1},
      static_cast<std::size_t>(
          std::ceil(alpha * static_cast<double>(v.size()))));
  const std::size_t k = support - 1;
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

static std::size_t ceil_capacity_saturated(double x);

static std::size_t resolution_frontier_capacity(
    std::vector<std::pair<double, double>> values,
    std::size_t q_base) {
  values.erase(
      std::remove_if(values.begin(), values.end(), [](const auto& value) {
        return !std::isfinite(value.first) || !std::isfinite(value.second) ||
               value.first <= 0.0 || value.second <= 0.0;
      }),
      values.end());
  if (values.empty() || q_base == 0) return q_base;
  std::sort(values.begin(), values.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });

  double total_materiality = 0.0;
  for (const auto& value : values) total_materiality += value.second;
  if (!(total_materiality > 0.0)) return q_base;

  std::size_t best_q = q_base;
  double best_utility = 0.0;
  double resolved_materiality = 0.0;
  std::size_t i = 0;
  while (i < values.size()) {
    const double q_value = values[i].first;
    while (i < values.size() && values[i].first == q_value) {
      resolved_materiality += values[i].second;
      ++i;
    }
    const std::size_t q = ceil_capacity_saturated(q_value);
    if (q <= q_base) continue;
    const double gain = resolved_materiality / total_materiality;
    const double cost =
        static_cast<double>(q - q_base) / static_cast<double>(q_base);
    const double utility = gain - cost;
    if (utility > best_utility) {
      best_utility = utility;
      best_q = q;
    }
  }
  return best_q;
}

static std::size_t geometric_midpoint(std::size_t lower, std::size_t upper) {
  if (lower >= upper) return upper;
  const long double product =
      static_cast<long double>(lower) * static_cast<long double>(upper);
  std::size_t midpoint =
      ceil_capacity_saturated(static_cast<double>(std::sqrt(product)));
  midpoint = std::max(midpoint, lower + 1);
  return std::min(midpoint, upper);
}

static std::size_t strict_resolution_capacity(double x) {
  if (!std::isfinite(x) || x < 0.0) return 0;
  const double max_q = static_cast<double>(std::numeric_limits<std::size_t>::max());
  if (x >= max_q) return std::numeric_limits<std::size_t>::max();
  // Certificate classification is strict: LB > tau or UB < tau. Therefore a
  // ratio that lands exactly on an integer needs one additional counter.
  return static_cast<std::size_t>(std::floor(x)) + 1;
}

static std::size_t ceil_capacity_saturated(double x) {
  if (!std::isfinite(x) || x <= 0.0) return 0;
  const double max_q = static_cast<double>(std::numeric_limits<std::size_t>::max());
  if (x >= max_q) return std::numeric_limits<std::size_t>::max();
  return static_cast<std::size_t>(std::ceil(x));
}

PolicyResult next_q_ss(const GlobalResultLB& R,
                       const std::vector<SnapshotEx>& snaps,
                       const PolicyConfig& cfg,
                       PolicyState* state) {
  PolicyResult out{};
  const std::size_t m = snaps.size();
  const std::size_t n_param = cfg.n_param ? cfg.n_param : 1;

  // Clamp helper
  auto clamp_q = [&](std::size_t q){
    if (cfg.q_cap != std::size_t(-1)) q = std::min(q, cfg.q_cap);
    q = std::max(q, cfg.q_min);
    if (cfg.q_max != std::size_t(-1)) q = std::min(q, cfg.q_max);
    return q;
  };

  if (cfg.kind == PolicyKind::fixed) {
    out.q_next = clamp_q(std::max<std::size_t>(n_param, cfg.q_cur ? cfg.q_cur : n_param));
    return out;
  }

  if (cfg.kind == PolicyKind::difficulty) {
    const std::size_t q_base = std::max<std::size_t>(n_param, cfg.q_cur ? cfg.q_cur : n_param);
    if (R.N_global == 0 || m == 0) {
      out.q_next = clamp_q(q_base);
      return out;
    }

    // Build key -> local candidate for O(1) reporter/eps lookup per key.
    std::vector<std::unordered_map<Id128, std::size_t, Id128Hash>> local_maps(m);
    for (std::size_t i = 0; i < m; ++i) {
      local_maps[i].reserve(snaps[i].candidates.size() * 2 + 8);
      for (std::size_t ci = 0; ci < snaps[i].candidates.size(); ++ci) {
        local_maps[i][snaps[i].candidates[ci].id] = ci;
      }
    }

    // Per-partition non-reporter minimum counter m_j^t.
    std::vector<double> min_counter(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
      if (snaps[i].q_local == 0 || snaps[i].candidates.size() < snaps[i].q_local) {
        min_counter[i] = 0.0;
        continue;
      }
      std::uint32_t mn = std::numeric_limits<std::uint32_t>::max();
      for (const auto& c : snaps[i].candidates) mn = std::min(mn, c.est);
      min_counter[i] = (mn == std::numeric_limits<std::uint32_t>::max()) ? 0.0 : static_cast<double>(mn);
    }

    const double N_total = static_cast<double>(R.N_global);
    std::vector<double> m_vals;
    std::vector<std::pair<double, double>> q_amb_materiality;
    const double threshold_rel = 1.0 / static_cast<double>(n_param);
    const std::size_t q_oper = std::max<std::size_t>(n_param, cfg.q_cur ? cfg.q_cur : n_param);
    const double rm = std::max(1e-12, cfg.r_m);

    for (const auto& gi : R.items) {
      if (cfg.candidate_ids && cfg.candidate_ids->find(gi.id) == cfg.candidate_ids->end()) continue;
      const bool is_cert_non_hh = (gi.cert_ub < R.threshold);
      if (is_cert_non_hh) continue; // T^t = K_+^t U K_?^t
      const bool ambiguous = (gi.cert_lb < R.threshold && gi.cert_ub >= R.threshold);

      double inflation = 0.0;
      double residual = 0.0;
      for (std::size_t i = 0; i < m; ++i) {
        const auto it = local_maps[i].find(gi.id);
        if (it != local_maps[i].end()) {
          const std::size_t ci = it->second;
          if (snaps[i].has_error_bounds() && ci < snaps[i].errors.size()) {
            inflation += static_cast<double>(snaps[i].errors[ci].eps);
          }
        }
        else residual += min_counter[i];
      }

      const double I = inflation / N_total;
      const double H = residual / N_total;
      const double Mv = std::max(I, H);
      m_vals.push_back(Mv);

      if (ambiguous) {
        const double p_hat = static_cast<double>(gi.est) / N_total;
        const double cert_lb = std::max(0.0, static_cast<double>(gi.cert_lb) / N_total);
        const double cert_ub = std::min(1.0, static_cast<double>(gi.cert_ub) / N_total);
        double side_uncertainty = 0.0;
        double side_margin = 0.0;
        if (p_hat >= threshold_rel) {
          side_uncertainty = std::max(0.0, p_hat - cert_lb);
          side_margin = std::max(0.0, p_hat - threshold_rel);
        } else {
          side_uncertainty = std::max(0.0, cert_ub - p_hat);
          side_margin = std::max(0.0, threshold_rel - p_hat);
        }
        // Under the standard inverse-capacity error model, uncertainty at q'
        // is side_uncertainty * q_oper / q'. The smallest finite q' that
        // strictly separates the key therefore obeys:
        //
        //   q' > q_oper * side_uncertainty / side_margin.
        //
        // Materiality is the estimate's distance from the threshold, measured
        // in threshold units. Near-ties therefore carry proportionally little
        // service weight even when their exact resolution demand is large.
        if (side_uncertainty > 0.0 && side_margin > 0.0) {
          const double q_amb_f =
              static_cast<double>(q_oper) * (side_uncertainty / side_margin);
          const std::size_t qk = strict_resolution_capacity(q_amb_f);
          if (qk > 0) {
            const double materiality = side_margin / threshold_rel;
            q_amb_materiality.emplace_back(
                static_cast<double>(std::max<std::size_t>(n_param, qk)),
                materiality);
          }
        }
      }
    }

    const double alpha = std::clamp(cfg.alpha_req, 0.0, 1.0);
    if (m_vals.empty()) {
      out.margin_alpha = 0.0;
    } else {
      std::vector<double> margin_work = m_vals;
      out.margin_alpha = quantile_inplace(margin_work, alpha);
    }
    out.service_violation = out.margin_alpha > rm;
    auto* st = state ? &state->difficulty : nullptr;
    const std::size_t q_cur = std::max<std::size_t>(n_param, q_oper);
    if (out.service_violation) {
      const double q_up_f =
          static_cast<double>(q_cur) * (out.margin_alpha / rm);
      out.q_up = std::max<std::size_t>(
          q_cur == std::numeric_limits<std::size_t>::max() ? q_cur : q_cur + 1,
          ceil_capacity_saturated(q_up_f));
    }

    std::size_t q_baseline = out.service_violation ? out.q_up : q_cur;
    if (st) {
      const std::size_t evidence_needed =
          std::max<std::size_t>(std::size_t{1}, cfg.residual_guard_window);
      if (st->probe_lower == 0) st->probe_lower = n_param;

      if (out.service_violation) {
        const bool failed_probe = st->probe_active;
        out.probe_failed = failed_probe;
        if (failed_probe) {
          st->probe_lower = std::max(
              st->probe_lower,
              q_cur == std::numeric_limits<std::size_t>::max()
                  ? q_cur
                  : q_cur + 1);
        } else {
          // A direct upward violation describes the current regime but says
          // nothing about the lower boundary learned by an earlier probe.
          st->probe_lower = n_param;
          st->guarded_demand = 0;
          st->probe_residual = 0.0;
        }
        q_baseline = std::max(st->probe_lower, out.q_up);
        if (cfg.probe_strategy == ProbeStrategy::pressure) {
          if (failed_probe) {
            const std::size_t failed_depth =
                std::max<std::size_t>(
                    std::size_t{1}, st->probe_retry_depth);
            st->probe_retry_depth =
                std::max<std::size_t>(
                    std::size_t{1}, (failed_depth + 1) / 2);
          } else {
            st->probe_retry_depth = 0;
          }
        }
        if (st->sufficient_upper > 0 && q_baseline > st->sufficient_upper) {
          st->sufficient_upper = 0;
        }
        if (cfg.probe_residual_guard && failed_probe) {
          st->guarded_demand = q_baseline;
          st->probe_residual =
              std::max(0.0, std::log(out.margin_alpha / rm));
        }
        st->sufficient_streak = 0;
        st->probe_success_streak = 0;
        st->probe_active = false;
      } else {
        if (st->sufficient_upper == 0) {
          st->sufficient_upper = q_cur;
        } else {
          st->sufficient_upper = std::min(st->sufficient_upper, q_cur);
        }

        if (cfg.probe_residual_guard && st->guarded_demand > 0 &&
            !st->probe_active) {
          q_baseline = q_cur;
          if (cfg.probe_strategy == ProbeStrategy::pressure) {
            st->probe_success_streak = 0;
            st->sufficient_streak = 0;
            st->guarded_demand = 0;
            st->probe_residual = 0.0;
          } else {
            ++st->probe_success_streak;
            if (st->probe_success_streak >= evidence_needed) {
              st->probe_success_streak = 0;
              st->sufficient_streak = 0;
              st->guarded_demand = 0;
              st->probe_residual = 0.0;
            }
          }
        } else if (st->probe_active && cfg.probe_residual_guard) {
          q_baseline = q_cur;
          if (cfg.probe_strategy == ProbeStrategy::pressure) {
            st->probe_active = false;
            st->probe_success_streak = 0;
            st->sufficient_streak = 0;
            st->probe_retry_depth = 0;
          } else {
            ++st->probe_success_streak;
            if (st->probe_success_streak >= evidence_needed) {
              st->probe_active = false;
              st->probe_success_streak = 0;
              st->sufficient_streak = 0;
            }
          }
        } else {
          if (st->probe_active) {
            st->probe_active = false;
            st->probe_success_streak = 0;
          }
          ++st->sufficient_streak;
          q_baseline = q_cur;

          const bool bracket_open =
              st->sufficient_upper > st->probe_lower + 1;
          const bool comfort_open =
              cfg.probe_strategy != ProbeStrategy::bracket && q_cur > n_param;
          const double pressure_ratio =
              std::clamp(out.margin_alpha / rm, 0.0, 1.0);
          const bool pressure_authorized =
              cfg.probe_strategy == ProbeStrategy::pressure &&
              pressure_ratio <=
                  std::clamp(cfg.probe_pressure_gate, 0.0, 1.0) &&
              q_cur > n_param;
          const bool fixed_window_ready =
              st->sufficient_streak >= evidence_needed;
          if (cfg.censored_control &&
              fixed_window_ready &&
              (cfg.probe_strategy != ProbeStrategy::pressure ||
               pressure_authorized) &&
              (bracket_open || comfort_open)) {
            std::size_t probe = q_cur;
            if (cfg.probe_strategy == ProbeStrategy::pressure) {
              if (pressure_ratio > 0.0) {
                // The pressure ratio authorizes a probe but does not establish
                // a safe lower capacity. Move only halfway to its modeled
                // boundary in linear capacity space.
                const std::size_t modeled_boundary = std::max<std::size_t>(
                    n_param,
                    ceil_capacity_saturated(
                        static_cast<double>(q_cur) * pressure_ratio));
                probe =
                    q_cur - (q_cur - modeled_boundary) / 2;
              } else {
                // A zero margin is censored at the deployed capacity. It
                // authorizes only a shallow half-n release, never a direct
                // return to the discoverability floor.
                const std::size_t shallow_release =
                    std::max<std::size_t>(std::size_t{1}, (n_param + 1) / 2);
                probe = q_cur > shallow_release
                    ? q_cur - shallow_release
                    : n_param;
                probe = std::max(probe, n_param);
              }
              const std::size_t maximum_release =
                  std::max<std::size_t>(std::size_t{1}, (n_param + 1) / 2);
              const std::size_t cautious_floor =
                  q_cur > maximum_release ? q_cur - maximum_release : n_param;
              probe = std::max(probe, cautious_floor);
              if (st->probe_retry_depth > 0 && probe < q_cur) {
                const std::size_t proposed_depth = q_cur - probe;
                probe = q_cur - std::min(
                    proposed_depth, st->probe_retry_depth);
              }
              if (cfg.probe_residual_guard && st->guarded_demand > 0) {
                probe = std::max(probe, st->guarded_demand);
              }
            } else if (cfg.probe_strategy == ProbeStrategy::comfort) {
              const double ratio =
                  std::clamp(out.margin_alpha / rm, 0.0, 1.0);
              const std::size_t supported_target = std::max<std::size_t>(
                  n_param,
                  ceil_capacity_saturated(
                      static_cast<double>(q_cur) * ratio));
              const std::size_t comfort_target =
                  q_cur - (q_cur - supported_target) / 2;
              const std::size_t maximum_release =
                  std::max<std::size_t>(std::size_t{1}, (n_param + 1) / 2);
              const std::size_t cautious_floor =
                  q_cur > maximum_release
                      ? q_cur - maximum_release
                      : n_param;
              probe = std::max({n_param, cautious_floor, comfort_target});
              if (cfg.probe_residual_guard && st->guarded_demand > 0) {
                probe = std::max(probe, st->guarded_demand);
              }
            } else {
              std::size_t lower = st->probe_lower;
              if (cfg.probe_residual_guard && st->guarded_demand > 0) {
                lower = std::max(lower, st->guarded_demand);
                lower = std::min(lower, st->sufficient_upper - 1);
              }
              probe = geometric_midpoint(lower, st->sufficient_upper);
            }
            if (probe < q_cur) {
              q_baseline = probe;
              out.probe_issued = true;
              st->probe_active = true;
              st->probe_success_streak = 0;
              if (cfg.probe_strategy == ProbeStrategy::pressure) {
                st->probe_retry_depth = q_cur - probe;
              }
            }
            st->sufficient_streak = 0;
          }
        }
      }
    }
    out.q_baseline = std::max<std::size_t>(n_param, q_baseline);
    out.z_hat =
        std::log(static_cast<double>(std::max<std::size_t>(1, out.q_baseline)));
    out.b_q = std::log(static_cast<double>(out.q_baseline) /
                       static_cast<double>(q_cur));
    if (st) st->b_req = out.b_q;

    // Build the observable ambiguity-resolution frontier relative to the
    // controller-selected baseline. Ambiguity already covered by q_B needs no
    // lift.
    std::vector<std::pair<double, double>> q_amb_frontier;
    q_amb_frontier.reserve(q_amb_materiality.size());
    for (const auto& [qk, materiality] : q_amb_materiality) {
      if (qk <= static_cast<double>(out.q_baseline)) continue;
      q_amb_frontier.emplace_back(qk, materiality);
    }
    const std::size_t q_amb =
        resolution_frontier_capacity(q_amb_frontier, out.q_baseline);

    // Attribute to ambiguity exactly its multiplicative lift over q_B, then
    // retain an unrefreshed margin with the same geometric decay used by the
    // ambiguity guard.
    const double observed_ambiguity_margin =
        (cfg.ambiguity_adjust && q_amb > out.q_baseline && out.q_baseline > 0)
            ? std::log(static_cast<double>(q_amb) /
                       static_cast<double>(out.q_baseline))
            : 0.0;
    const double beta = std::clamp(cfg.residual_guard_decay, 0.0, 1.0);
    const double ambiguity_margin = cfg.ambiguity_adjust
        ? std::max(observed_ambiguity_margin,
                   beta * (st ? st->ambiguity_margin : 0.0))
        : 0.0;
    const std::size_t q_amb_pred = cfg.ambiguity_adjust
        ? std::max<std::size_t>(
              n_param,
              ceil_capacity_saturated(
                  static_cast<double>(out.q_baseline) *
                  std::exp(ambiguity_margin)))
        : n_param;

    if (cfg.ambiguity_adjust) {
      out.q_pred = std::max<std::size_t>(out.q_baseline, q_amb_pred);
    } else {
      out.q_pred = out.q_baseline;
    }
    out.q_req_unclipped = out.q_up;
    out.q_req = out.q_baseline;
    out.q_base = std::max(out.q_baseline, q_amb);
    out.q_pred_tilde = static_cast<double>(out.q_pred);

    if (st) st->ambiguity_margin = ambiguity_margin;
    out.g_amb = ambiguity_margin;

    out.q_next = clamp_q(std::max<std::size_t>(n_param, out.q_pred));
    return out;
  }

  // Fallback
  out.q_next = clamp_q(std::max<std::size_t>(n_param, cfg.q_cur ? cfg.q_cur : n_param));
  return out;
}

} // namespace sizing
} // namespace hh
