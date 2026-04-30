#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <deque>
#include "hh/coord/coordinator.hpp"   // GlobalResultLB
#include "hh/sketches/isketch.hpp"    // SnapshotEx
#include "hh/core/id128.hpp"          // Id128, Id128Hash

namespace hh {
namespace sizing {

// Supported sizing policies:
//  - fixed: hold q constant (still clamped by q_min/q_max)
//  - difficulty: requirement-based predictive controller from the paper.
enum class PolicyKind { fixed, difficulty };

struct PolicyConfig {
  PolicyKind kind{PolicyKind::fixed};

  // Common
  std::size_t n_param{0};    // global HH denominator n (strict HH threshold = floor(N_global/n)+1)
  double r{0.10};            // retained for compatibility with older configs; unused by difficulty
  std::size_t q_cap{std::size_t(-1)};  // per-partition memory cap (in entries)
  std::size_t q_cur{0};      // current q_t (for reporting)

  double alpha_req{0.8};     // quantile for Q_alpha in q_Req^t under difficulty policy

  // Optional minimum/maximum clamps
  std::size_t q_min{1};
  std::size_t q_max{std::size_t(-1)};

  // Optional candidate filter (paper: sizing over candidate set C_t)
  const std::unordered_set<Id128, Id128Hash>* candidate_ids{nullptr};

  // EWMA + safety calibration knobs for difficulty-aware sizing policy
  double rho{0.5};                  // EWMA mixing for effective-requirement forecast z_eff
  double rho_a{0.5};                // EWMA mixing for ambiguity level \bar D_A
  std::size_t calib_window{0};       // residual-history length (0 => use all collected up to cap)

  // Calibrated safety level in the theory: b_eff = quantile_{1-delta}(effective residuals)
  double delta_m{0.1};              // used as \delta for effective residual calibration
  std::size_t trend_h{3};           // ambiguity-trend horizon h >= 2
  double lambda_a{0.0};             // ambiguity-level gain \lambda_A
  double lambda_g{0.0};             // ambiguity-trend gain \lambda_G

  // Sizing coefficients for difficulty policy
  double r_m{0.10};
};

struct DifficultyState {
  double z_eff_hat{NAN};       // one-step forecast \hat z_eff
  double da_smooth{NAN};       // smoothed ambiguity \bar D_A
  std::deque<double> z_eff_res_hist; // residuals R_eff for calibration quantile
  std::deque<double> da_smooth_hist; // recent \bar D_A for trend slope G_A
};

struct PolicyState {
  DifficultyState difficulty;
};

struct PolicyResult {
  std::size_t q_next{0};
  // debug/telemetry
  double da{NAN};          // for difficulty: ambiguity
  double z_hat{NAN};       // forecast of z_eff
  double da_hat{NAN};      // smoothed ambiguity \bar D_A
  double b_q{NAN};         // effective residual calibration bias b_eff
  double eta_a{NAN};       // ambiguity trend G_A
  double q_pred_tilde{NAN}; // calibrated predictor \tilde q_eff^{t+1}
  double da_tilde{NAN};    // retained for compatibility (unused)
  double p_plus{NAN};      // retained for compatibility (unused by v2)
  double g_amb{NAN};       // retained for compatibility (unused by v2)
  double g_head{NAN};      // retained for compatibility (unused by v2)
  double kappa_t{NAN};     // retained for compatibility (unused by v2)
  std::size_t q_req{0};   // current-window realized required capacity
  std::size_t q_pred{0};  // calibrated predictive next-window effective capacity
  std::size_t q_base{0};  // effective requirement replay ceil(exp(z_eff))
  std::size_t q_tar{0};   // retained for compatibility (same as q_pred in v2)
};

/// Compute next q for SpaceSaving given the global reduction and all local snapshots.
PolicyResult next_q_ss(const GlobalResultLB& R,
                       const std::vector<SnapshotEx>& snaps,
                       const PolicyConfig& cfg,
                       PolicyState* state = nullptr);

} // namespace sizing
} // namespace hh
