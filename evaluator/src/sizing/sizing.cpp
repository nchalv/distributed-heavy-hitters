#include "hh/sizing/sizing.hpp"
#include <algorithm>
#include <limits>

namespace hh {
namespace sizing {

static double quantile_inplace(std::vector<double>& v, double alpha) {
  if (v.empty()) return 0.0;
  if (alpha <= 0.0) return *std::min_element(v.begin(), v.end());
  if (alpha >= 1.0) return *std::max_element(v.begin(), v.end());
  const std::size_t k = static_cast<std::size_t>(std::floor(alpha * (v.size()-1)));
  std::nth_element(v.begin(), v.begin()+k, v.end());
  return v[k];
}

static double upper_order_stat(const std::deque<double>& hist, double delta, std::size_t cap_window) {
  if (hist.empty()) return 1.0;
  const std::size_t use = (cap_window > 0) ? std::min<std::size_t>(cap_window, hist.size()) : hist.size();
  std::vector<double> work;
  work.reserve(use);
  const std::size_t start = hist.size() - use;
  for (std::size_t i = start; i < hist.size(); ++i) {
    if (std::isfinite(hist[i])) work.push_back(hist[i]);
  }
  if (work.empty()) return 1.0;
  const double d = std::clamp(delta, 0.0, 1.0);
  const std::size_t L = work.size();
  std::size_t k = static_cast<std::size_t>(std::ceil((static_cast<double>(L) + 1.0) * (1.0 - d)));
  k = std::max<std::size_t>(1, std::min<std::size_t>(k, L));
  std::nth_element(work.begin(), work.begin() + (k - 1), work.end());
  return work[k - 1];
}

static double upper_order_stat_or(const std::deque<double>& hist,
                                  double delta,
                                  std::size_t cap_window,
                                  double fallback) {
  if (hist.empty()) return fallback;
  return upper_order_stat(hist, delta, cap_window);
}

static double ambiguity_trend_slope(const std::deque<double>& hist, std::size_t h) {
  if (h < 2 || hist.size() < h) return 0.0;
  const std::size_t start = hist.size() - h;
  const double ell_bar = static_cast<double>(h - 1) / 2.0;
  double num = 0.0;
  double den = 0.0;
  for (std::size_t ell = 0; ell < h; ++ell) {
    const double x = static_cast<double>(ell) - ell_bar;
    const double y = hist[start + ell];
    num += x * y;
    den += x * x;
  }
  return (den > 0.0) ? (num / den) : 0.0;
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
    std::vector<double> q_req_vals;
    double sum_hat_target = 0.0;
    double sum_hat_ambiguous = 0.0;

    for (const auto& gi : R.items) {
      if (cfg.candidate_ids && cfg.candidate_ids->find(gi.id) == cfg.candidate_ids->end()) continue;
      const bool is_cert_non_hh = (gi.cert_ub < R.threshold);
      if (is_cert_non_hh) continue; // T^t = K_+^t U K_?^t
      const bool ambiguous = (gi.cert_lb < R.threshold && gi.cert_ub >= R.threshold);
      const double p_hat = static_cast<double>(gi.est) / N_total;
      sum_hat_target += p_hat;
      if (ambiguous) sum_hat_ambiguous += p_hat;

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
    }

    out.da = (sum_hat_target > 0.0) ? (sum_hat_ambiguous / sum_hat_target) : 0.0; // D_A^t

    // Per-key required capacity:
    // q_req(k) = max{ n, ceil( q_oper * M(k) / r_m ) }.
    const double rm = std::max(1e-12, cfg.r_m);
    const std::size_t q_oper = std::max<std::size_t>(n_param, cfg.q_cur ? cfg.q_cur : n_param);
    q_req_vals.reserve(m_vals.size());
    for (double mk : m_vals) {
      const double qk_f = static_cast<double>(q_oper) * (mk / rm);
      const std::size_t qk = std::max<std::size_t>(n_param, static_cast<std::size_t>(std::ceil(qk_f)));
      q_req_vals.push_back(static_cast<double>(qk));
    }
    // Window-level required capacity:
    // q_Req^t = Q_alpha({ q_req(k) : k in T^t }).
    const double qreq_q = q_req_vals.empty() ? static_cast<double>(n_param)
                                             : quantile_inplace(q_req_vals, cfg.alpha_req);
    out.q_req = std::max<std::size_t>(n_param, static_cast<std::size_t>(std::ceil(qreq_q)));

    auto* st = state ? &state->difficulty : nullptr;

    // Smoothed ambiguity level \bar D_A^t.
    const double rho_a = std::clamp(cfg.rho_a, 0.0, 1.0 - 1e-12);
    const double da_prev_s = (st && std::isfinite(st->da_smooth)) ? st->da_smooth : out.da;
    out.da_hat = (1.0 - rho_a) * out.da + rho_a * da_prev_s;

    if (st) {
      st->da_smooth = out.da_hat;
      st->da_smooth_hist.push_back(out.da_hat);
      if (cfg.calib_window > 0) {
        // Keep enough history for both calibration and ambiguity trend.
        const std::size_t hist_cap = std::max<std::size_t>(
            cfg.calib_window,
            std::max<std::size_t>(2, cfg.trend_h));
        while (st->da_smooth_hist.size() > hist_cap) st->da_smooth_hist.pop_front();
      }
    }

    // Signed ambiguity trend G_A^t over last h smoothed values.
    const std::size_t h = std::max<std::size_t>(2, cfg.trend_h);
    out.eta_a = st ? ambiguity_trend_slope(st->da_smooth_hist, h) : 0.0;

    // Effective requirement in log-space:
    // z_eff^t = log(q_req^t) + lambda_A * \bar D_A^t + lambda_G * G_A^t.
    const double z_eff_t =
        std::log(static_cast<double>(std::max<std::size_t>(1, out.q_req)))
        + cfg.lambda_a * out.da_hat
        + cfg.lambda_g * out.eta_a;

    // One-step forecast \hat z_{t+1|t} (EWMA of effective signal).
    const double rho = std::clamp(cfg.rho, 0.0, 1.0 - 1e-12);
    const double z_prev_hat = (st && std::isfinite(st->z_eff_hat)) ? st->z_eff_hat : z_eff_t;
    out.z_hat = (1.0 - rho) * z_eff_t + rho * z_prev_hat;

    // Residual R_eff^t = z_eff^t - \hat z_eff_{t|t-1}, then calibrated upper bias b_eff.
    if (st) {
      st->z_eff_res_hist.push_back(z_eff_t - z_prev_hat);
      if (cfg.calib_window > 0) {
        while (st->z_eff_res_hist.size() > cfg.calib_window) st->z_eff_res_hist.pop_front();
      }
      st->z_eff_hat = out.z_hat;
    }
    out.b_q = upper_order_stat_or(st ? st->z_eff_res_hist : std::deque<double>{},
                                  cfg.delta_m, cfg.calib_window, 0.0);
    out.b_q = std::max(0.0, out.b_q); // one-sided calibration per paper

    // Calibrated predictive effective requirement \tilde q_eff^{t+1}.
    out.q_pred_tilde = static_cast<double>(
        std::max<std::size_t>(n_param,
          static_cast<std::size_t>(std::ceil(std::exp(out.z_hat + out.b_q)))));
    out.q_pred = static_cast<std::size_t>(out.q_pred_tilde);

    // Reactive effective replay q_eff^t and predictive actuation.
    out.q_base = std::max<std::size_t>(n_param,
               static_cast<std::size_t>(std::ceil(std::exp(z_eff_t))));
    out.q_tar = out.q_pred;
    out.da_tilde = NAN;
    out.p_plus = NAN;
    out.g_amb = NAN;
    out.g_head = NAN;
    out.kappa_t = NAN;

    out.q_next = clamp_q(std::max<std::size_t>(n_param, out.q_pred));
    return out;
  }

  // Fallback
  out.q_next = clamp_q(std::max<std::size_t>(n_param, cfg.q_cur ? cfg.q_cur : n_param));
  return out;
}

} // namespace sizing
} // namespace hh
