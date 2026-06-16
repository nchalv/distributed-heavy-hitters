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
//  - difficulty: certificate-pressure controller from the paper.
enum class PolicyKind { fixed, difficulty };
enum class ProbeStrategy { bracket, comfort, pressure };

struct PolicyConfig {
  PolicyKind kind{PolicyKind::fixed};

  // Common
  std::size_t n_param{0};    // global HH denominator n (strict HH threshold = floor(N_global/n)+1)
  double r{0.10};            // retained for compatibility with older configs; unused by difficulty
  std::size_t q_cap{std::size_t(-1)};  // per-partition memory cap (in entries)
  std::size_t q_cur{0};      // current q_t (for reporting)

  double alpha_req{0.8};     // service quantile for M_alpha under difficulty policy

  // Optional minimum/maximum clamps
  std::size_t q_min{1};
  std::size_t q_max{std::size_t(-1)};

  // Optional candidate filter (paper: sizing over candidate set C_t)
  const std::unordered_set<Id128, Id128Hash>* candidate_ids{nullptr};

  // Number of consecutive sufficient observations required before a downward
  // probe is issued or a guarded probe is accepted.
  std::size_t residual_guard_window{2};
  // Shared geometric decay for the independent ambiguity margin.
  double residual_guard_decay{0.5};
  // Deprecated compatibility switch; one-sided pressure control always reacts
  // immediately to demonstrated insufficiency.
  bool symmetric_relaxation{false};
  // Enable physical downward probing after repeated sufficient observations.
  // When false, the controller is upward-only and never releases memory.
  bool censored_control{true};
  // Retain the upward demand exposed by a failed probe while recovery is
  // confirmed. The guard clears after two sufficient recovery observations.
  bool probe_residual_guard{true};
  // Select lower probes from the historical bracket, current quantile margin
  // headroom, or guarded margin-comfort probing with a pressure veto.
  ProbeStrategy probe_strategy{ProbeStrategy::bracket};
  double probe_pressure_gate{0.5}; // maximum M_alpha/r_M authorizing release
  // Retained for compatibility with older method specs; ignored by difficulty.
  double delta_m{0.1};
  bool ambiguity_adjust{true};      // if true, select the ambiguity-resolution frontier knee

  // Sizing coefficients for difficulty policy
  double r_m{0.10};
};

struct DifficultyState {
  std::size_t probe_lower{0}; // lower boundary established by failed probes
  std::size_t sufficient_upper{0}; // last capacity observed sufficient
  std::size_t sufficient_streak{0}; // consecutive sufficient observations
  bool probe_active{false}; // current capacity was selected as a downward probe
  std::size_t guarded_demand{0}; // upward demand exposed by the latest failed probe
  double probe_residual{0.0}; // positive log(M_alpha/r_m) at the latest failure
  std::size_t probe_success_streak{0};
  std::size_t probe_retry_depth{0}; // maximum release after a failed probe
  double b_req{0.0};           // log baseline movement relative to current q
  double ambiguity_margin{0.0}; // persisted ambiguity margin a_Amb in log-capacity space
};

struct PolicyState {
  DifficultyState difficulty;
};

struct PolicyResult {
  std::size_t q_next{0};
  // debug/telemetry
  double margin_alpha{NAN}; // realized service margin M_alpha
  bool service_violation{false};
  std::size_t q_up{0};     // upward demand; zero when the service target is met
  std::size_t q_baseline{0}; // target selected by upward response or probing
  bool probe_issued{false};
  bool probe_failed{false};
  double z_hat{NAN};       // log baseline target
  double b_q{NAN};         // log movement of baseline target relative to q_cur
  double q_pred_tilde{NAN}; // calibrated predictive next-window capacity after ambiguity guard
  double g_amb{NAN};       // ambiguity-attributable log-capacity margin
  // Deprecated aliases retained while experiment/reporting code migrates.
  std::size_t q_req_unclipped{0}; // q_up, or zero when no upward demand exists
  std::size_t q_req{0};   // baseline next-window target
  std::size_t q_pred{0};  // calibrated predictive next-window effective capacity
  std::size_t q_base{0};  // ambiguity-adjusted target
};

/// Compute next q for SpaceSaving given the global reduction and all local snapshots.
PolicyResult next_q_ss(const GlobalResultLB& R,
                       const std::vector<SnapshotEx>& snaps,
                       const PolicyConfig& cfg,
                       PolicyState* state = nullptr);

} // namespace sizing
} // namespace hh
