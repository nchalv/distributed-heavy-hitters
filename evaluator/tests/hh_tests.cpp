#include "hh/coord/coordinator.hpp"
#include "hh/core/arena_map.hpp"
#include "hh/core/hash.hpp"
#include "hh/hybrid/hybrid.hpp"
#include "hh/io/reader.hpp"
#include "hh/sizing/sizing.hpp"
#include "hh/sketches/ss.hpp"

#include <zstr.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using hh::Id128;

struct TestFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw TestFailure(message);
}

void init_hashing() {
  std::array<std::uint8_t, 32> kid{};
  std::array<std::uint8_t, 32> kfp{};
  std::array<std::uint8_t, 32> kidx{};
  for (std::size_t i = 0; i < kid.size(); ++i) {
    kid[i] = static_cast<std::uint8_t>(i + 1);
    kfp[i] = static_cast<std::uint8_t>(101 + i);
    kidx[i] = static_cast<std::uint8_t>(201 + i);
  }
  hh::set_secret_keys(kid, kfp, kidx);
}

Id128 id_for(const std::string& key) {
  return hh::id128_for(key);
}

const hh::GlobalItemLB* find_item(const hh::GlobalResultLB& result, const Id128& id) {
  for (const auto& item : result.items) {
    if (item.id == id) return &item;
  }
  return nullptr;
}

void write_gzip_json(const std::filesystem::path& path, const std::string& json) {
  std::ofstream fout(path, std::ios::binary);
  require(static_cast<bool>(fout), "failed to open gzip fixture for writing: " + path.string());
  zstr::ostream zout(fout);
  zout << json;
}

void test_space_saving_bounds_and_reconfigure() {
  hh::SpaceSaving ss(3);
  ss.update(id_for("a"), 5);
  ss.update(id_for("b"), 3);
  ss.update(id_for("c"), 2);
  ss.update(id_for("d"), 1);

  const auto snap = ss.snapshot_ex();
  require(snap.N_local == 11, "SpaceSaving tracks local mass");
  require(snap.q_local == 3, "SpaceSaving reports capacity");
  require(snap.candidates.size() <= 3, "SpaceSaving never reports more than q candidates");
  require(snap.has_error_bounds(), "SpaceSaving exports per-item error bounds");

  std::uint64_t counter_sum = 0;
  for (std::size_t i = 0; i < snap.candidates.size(); ++i) {
    require(snap.errors[i].lb <= snap.candidates[i].est, "candidate lower bound is not above estimate");
    require(snap.errors[i].eps <= snap.candidates[i].est, "candidate epsilon is not above estimate");
    counter_sum += snap.candidates[i].est;
  }
  require(counter_sum == snap.N_local, "SpaceSaving counters sum to observed mass");

  ss.reconfigure(2);
  const auto after_resize = ss.snapshot_ex();
  require(after_resize.N_local == 0, "reconfigure starts a fresh window");
  require(after_resize.q_local == 2, "reconfigure updates capacity");
  require(after_resize.candidates.empty(), "reconfigure clears resident candidates");

  ss.update(id_for("a"), 70000);
  const auto wide = ss.snapshot_ex();
  require(wide.N_local == 70000, "SpaceSaving supports counts beyond uint16 compact storage");
  require(wide.candidates.size() == 1 && wide.candidates[0].est == 70000,
          "wide-count promotion preserves large counts");
}

hh::GlobalResultLB make_manual_reduction() {
  const Id128 a = id_for("a");
  const Id128 b = id_for("b");
  const Id128 c = id_for("c");

  hh::SnapshotEx s0;
  s0.N_local = 10;
  s0.q_local = 2;
  s0.candidates = {{a, 6}, {b, 4}};
  s0.errors = {{6, 0}, {4, 0}};

  hh::SnapshotEx s1;
  s1.N_local = 10;
  s1.q_local = 2;
  s1.candidates = {{b, 4}, {c, 6}};
  s1.errors = {{4, 0}, {6, 0}};

  hh::ArenaMap m0;
  hh::ArenaMap m1;
  m0.bind_bytes(a, "a");
  m0.bind_bytes(b, "b");
  m1.bind_bytes(c, "c");

  std::vector<hh::SnapshotEx> snaps{s0, s1};
  std::vector<const hh::ArenaMap*> maps{&m0, &m1};
  return hh::Coordinator::reduce_global_with_lb(snaps, maps, 3);
}

void test_coordinator_certification_envelope() {
  const auto result = make_manual_reduction();
  require(result.N_global == 20, "coordinator sums global mass");
  require(result.threshold == 7, "coordinator uses strict floor(N/n)+1 threshold");
  require(result.items.size() == 3, "coordinator aggregates all reported ids");

  const auto* a = find_item(result, id_for("a"));
  const auto* b = find_item(result, id_for("b"));
  const auto* c = find_item(result, id_for("c"));
  require(a && b && c, "coordinator result contains expected keys");

  require(a->key == "a", "coordinator resolves keys through arena maps");
  require(a->est == 6 && a->cert_lb == 6 && a->cert_ub == 10,
          "coordinator adds non-reporter min counter to upper bound");
  require(!a->guaranteed, "key below certified threshold is not guaranteed");

  require(b->est == 8 && b->cert_lb == 8 && b->cert_ub == 8,
          "coordinator keeps fully reported exact key tight");
  require(b->guaranteed, "key with certified lower bound above threshold is guaranteed");

  require(c->est == 6 && c->cert_lb == 6 && c->cert_ub == 10,
          "coordinator computes symmetric hidden mass case");
}

void test_sizing_policy_outputs_are_clamped_and_finite() {
  const hh::sizing::PolicyConfig default_cfg;
  require(default_cfg.censored_control &&
              default_cfg.probe_residual_guard &&
              default_cfg.residual_guard_window == 2,
          "censored residual-guarded control is the default sizing policy");

  const auto result = make_manual_reduction();

  hh::SnapshotEx s0;
  s0.N_local = 10;
  s0.q_local = 2;
  s0.candidates = {{id_for("a"), 6}, {id_for("b"), 4}};
  s0.errors = {{6, 0}, {4, 0}};

  hh::SnapshotEx s1;
  s1.N_local = 10;
  s1.q_local = 2;
  s1.candidates = {{id_for("b"), 4}, {id_for("c"), 6}};
  s1.errors = {{4, 0}, {6, 0}};

  hh::sizing::PolicyConfig cfg;
  cfg.kind = hh::sizing::PolicyKind::difficulty;
  cfg.n_param = 3;
  cfg.q_cur = 2;
  cfg.q_min = 2;
  cfg.q_max = 40;
  cfg.q_cap = 40;
  cfg.alpha_req = 1.0;
  cfg.r_m = 0.25;

  hh::sizing::PolicyState state;
  const auto first = hh::sizing::next_q_ss(result, {s0, s1}, cfg, &state);
  require(first.q_req >= cfg.n_param, "difficulty policy reports a requirement at least n");
  require(first.q_next >= cfg.q_min && first.q_next <= cfg.q_cap && first.q_next <= cfg.q_max,
          "difficulty policy clamps q_next");
  require(first.q_base >= first.q_req,
          "ambiguity adjustment cannot lower the effective requirement");
  require(first.q_base == first.q_req,
          "ambiguity remains inactive when all resolution demands have negative utility");

  require(first.g_amb == 0.0,
          "non-binding ambiguity reports no actuation lift");
  require(first.q_pred >= cfg.n_param, "difficulty policy exports predictive effective capacity");

  cfg.alpha_req = 0.5;
  cfg.ambiguity_adjust = false;
  const auto unadjusted = hh::sizing::next_q_ss(result, {s0, s1}, cfg, nullptr);
  require(unadjusted.q_base == unadjusted.q_req,
          "requirement-only ablation keeps q_eff equal to q_req");

  const Id128 cheap_id = id_for("cheap-ambiguity");
  const Id128 costly_id = id_for("costly-ambiguity");
  hh::GlobalResultLB efficiency_result;
  efficiency_result.N_global = 1000;
  efficiency_result.threshold = 101;
  efficiency_result.items = {
      {cheap_id, "cheap-ambiguity", 110, 99, 1, 1.0, 99, 120, false},
      {costly_id, "costly-ambiguity", 101, 0, 1, 1.0, 0, 200, false},
  };
  hh::SnapshotEx efficiency_snap;
  efficiency_snap.N_local = 1000;
  efficiency_snap.q_local = 10;
  efficiency_snap.candidates = {{cheap_id, 110}, {costly_id, 101}};
  efficiency_snap.errors = {{110, 0}, {101, 0}};

  hh::sizing::PolicyConfig efficiency_cfg;
  efficiency_cfg.kind = hh::sizing::PolicyKind::difficulty;
  efficiency_cfg.n_param = 10;
  efficiency_cfg.q_cur = 10;
  efficiency_cfg.q_min = 10;
  efficiency_cfg.q_max = 2000;
  efficiency_cfg.q_cap = 2000;
  efficiency_cfg.alpha_req = 0.95;
  efficiency_cfg.r_m = 0.01;
  efficiency_cfg.ambiguity_adjust = true;
  const auto efficient = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, efficiency_cfg, nullptr);
  require(efficient.q_req == 10,
          "zero certificate inflation leaves the baseline requirement at n");
  require(efficient.q_base == 12,
          "cost-sensitive ambiguity serves the efficient resolution before the costly near-tie");
  require(std::abs(efficient.g_amb - std::log(12.0 / 10.0)) < 1e-12,
          "ambiguity reports its attributable log-capacity margin");

  hh::SnapshotEx high_error_snap = efficiency_snap;
  high_error_snap.errors = {{110, 100}, {101, 100}};
  hh::sizing::PolicyConfig upward_cfg = efficiency_cfg;
  upward_cfg.ambiguity_adjust = false;
  upward_cfg.censored_control = false;
  upward_cfg.probe_residual_guard = false;
  upward_cfg.q_cur = 40;
  const auto held = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, upward_cfg, nullptr);
  require(!held.service_violation && held.q_up == 0 &&
              held.q_baseline == 40 && held.q_pred == 40,
          "upward-only control holds capacity after a sufficient window");

  upward_cfg.q_cur = 10;
  const auto raised = hh::sizing::next_q_ss(
      efficiency_result, {high_error_snap}, upward_cfg, nullptr);
  require(raised.service_violation && raised.q_up == 100 &&
              raised.q_baseline == 100 && raised.q_pred == 100,
          "a service violation activates origin-aware upward sizing");

  hh::sizing::PolicyConfig probing_cfg = upward_cfg;
  probing_cfg.censored_control = true;
  probing_cfg.residual_guard_window = 2;
  probing_cfg.q_cur = 40;
  hh::sizing::PolicyState probing_state;
  const auto first_sufficient = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, probing_cfg, &probing_state);
  require(first_sufficient.q_pred == 40 && !first_sufficient.probe_issued,
          "one sufficient observation does not release memory");
  const auto second_sufficient = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, probing_cfg, &probing_state);
  require(second_sufficient.q_pred == 20 &&
              second_sufficient.probe_issued &&
              probing_state.difficulty.probe_active,
          "repeated sufficiency starts a physical geometric probe");

  probing_cfg.q_cur = 20;
  const auto successful_probe = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, probing_cfg, &probing_state);
  require(!successful_probe.service_violation &&
              successful_probe.q_pred == 20 &&
              !probing_state.difficulty.probe_active &&
              probing_state.difficulty.sufficient_upper == 20,
          "a sufficient unguarded probe becomes the retained upper point");

  hh::sizing::PolicyState failed_probe_state;
  probing_cfg.q_cur = 40;
  (void)hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, probing_cfg, &failed_probe_state);
  (void)hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, probing_cfg, &failed_probe_state);
  hh::SnapshotEx moderate_error_snap = efficiency_snap;
  moderate_error_snap.errors = {{110, 15}, {101, 15}};
  probing_cfg.q_cur = 20;
  const auto failed_probe = hh::sizing::next_q_ss(
      efficiency_result, {moderate_error_snap}, probing_cfg, &failed_probe_state);
  require(failed_probe.service_violation && failed_probe.probe_failed &&
              failed_probe.q_up == 30 && failed_probe.q_pred == 30 &&
              !failed_probe_state.difficulty.probe_active,
          "an insufficient probe recovers through fresh upward sizing");

  hh::sizing::PolicyConfig guarded_cfg = probing_cfg;
  guarded_cfg.probe_residual_guard = true;
  guarded_cfg.q_cur = 40;
  hh::sizing::PolicyState guarded_state;
  (void)hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, guarded_cfg, &guarded_state);
  (void)hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, guarded_cfg, &guarded_state);
  guarded_cfg.q_cur = 20;
  const auto guarded_failure = hh::sizing::next_q_ss(
      efficiency_result, {moderate_error_snap}, guarded_cfg, &guarded_state);
  require(guarded_failure.q_pred == 30 &&
              guarded_failure.probe_failed &&
              guarded_state.difficulty.guarded_demand == 30 &&
              guarded_state.difficulty.probe_residual > 0.0,
          "a failed probe retains its recovered demand");

  guarded_cfg.q_cur = 30;
  const auto first_recovery = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, guarded_cfg, &guarded_state);
  require(first_recovery.q_pred == 30 &&
              !first_recovery.probe_issued &&
              guarded_state.difficulty.guarded_demand == 30,
          "one sufficient recovery window holds the guarded demand");

  hh::SnapshotEx low_error_snap = efficiency_snap;
  low_error_snap.errors = {{110, 5}, {101, 5}};
  const auto confirmed_recovery = hh::sizing::next_q_ss(
      efficiency_result, {low_error_snap}, guarded_cfg, &guarded_state);
  require(confirmed_recovery.q_pred == 30 &&
              !confirmed_recovery.probe_issued &&
              guarded_state.difficulty.guarded_demand == 0,
          "two sufficient recovery windows clear the failed-probe guard");

  const auto first_post_recovery = hh::sizing::next_q_ss(
      efficiency_result, {low_error_snap}, guarded_cfg, &guarded_state);
  require(first_post_recovery.q_pred == 30 &&
              !first_post_recovery.probe_issued,
          "one post-recovery observation does not yet probe");
  const auto guarded_retry = hh::sizing::next_q_ss(
      efficiency_result, {low_error_snap}, guarded_cfg, &guarded_state);
  require(guarded_retry.q_pred == 26 && guarded_retry.probe_issued &&
              guarded_state.difficulty.probe_active,
          "confirmed recovery permits a later real probe");

  guarded_cfg.q_cur = 26;
  const auto guarded_first_success = hh::sizing::next_q_ss(
      efficiency_result, {low_error_snap}, guarded_cfg, &guarded_state);
  require(guarded_first_success.q_pred == 26 &&
              guarded_state.difficulty.probe_active &&
              guarded_state.difficulty.probe_success_streak == 1,
          "one successful guarded probe awaits confirmation");
  const auto guarded_second_success = hh::sizing::next_q_ss(
      efficiency_result, {low_error_snap}, guarded_cfg, &guarded_state);
  require(guarded_second_success.q_pred == 26 &&
              !guarded_state.difficulty.probe_active &&
              guarded_state.difficulty.probe_residual == 0.0,
          "two successful guarded observations accept the lower capacity");

  hh::sizing::PolicyState blocked_comfort_state;
  blocked_comfort_state.difficulty.sufficient_upper = 30;
  blocked_comfort_state.difficulty.sufficient_streak = 1;
  blocked_comfort_state.difficulty.guarded_demand = 30;
  blocked_comfort_state.difficulty.probe_success_streak = 0;
  hh::sizing::PolicyConfig blocked_comfort_cfg = guarded_cfg;
  blocked_comfort_cfg.probe_strategy = hh::sizing::ProbeStrategy::comfort;
  blocked_comfort_cfg.q_cur = 30;
  const auto blocked_comfort = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, blocked_comfort_cfg,
      &blocked_comfort_state);
  require(blocked_comfort.q_pred == 30 &&
              !blocked_comfort.probe_issued,
          "a blocking guard holds capacity instead of forcing a q-minus-one probe");

  hh::sizing::PolicyConfig comfort_cfg = probing_cfg;
  comfort_cfg.probe_strategy = hh::sizing::ProbeStrategy::comfort;
  comfort_cfg.probe_residual_guard = true;
  comfort_cfg.q_cur = 40;
  hh::sizing::PolicyState comfort_state;
  (void)hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, comfort_cfg, &comfort_state);
  const auto comfort_probe = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, comfort_cfg, &comfort_state);
  require(comfort_probe.q_pred == 35 && comfort_probe.probe_issued,
          "comfort probing uses a capped arithmetic midpoint");

  hh::sizing::PolicyConfig pressure_cfg = probing_cfg;
  pressure_cfg.probe_strategy = hh::sizing::ProbeStrategy::pressure;
  pressure_cfg.probe_residual_guard = true;
  pressure_cfg.q_cur = 40;

  hh::SnapshotEx hold_snap = efficiency_snap;
  hold_snap.errors = {{110, 7}, {101, 7}};
  hh::sizing::PolicyState hold_state;
  (void)hh::sizing::next_q_ss(
      efficiency_result, {hold_snap}, pressure_cfg, &hold_state);
  const auto pressure_hold = hh::sizing::next_q_ss(
      efficiency_result, {hold_snap}, pressure_cfg, &hold_state);
  require(pressure_hold.q_pred == 40 &&
              !pressure_hold.probe_issued &&
              !pressure_hold.service_violation,
          "pressure above one half of the budget vetoes a guarded probe");

  hh::sizing::PolicyState pressure_state;
  const auto pressure_wait = hh::sizing::next_q_ss(
      efficiency_result, {low_error_snap}, pressure_cfg, &pressure_state);
  require(pressure_wait.q_pred == 40 &&
              !pressure_wait.probe_issued,
          "one low-pressure observation does not bypass the temporal guard");
  const auto pressure_probe = hh::sizing::next_q_ss(
      efficiency_result, {low_error_snap}, pressure_cfg, &pressure_state);
  require(pressure_probe.q_pred == 35 &&
              pressure_probe.probe_issued,
          "guarded half-budget pressure uses a capped arithmetic midpoint");

  hh::sizing::PolicyState zero_state;
  (void)hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, pressure_cfg, &zero_state);
  const auto zero_probe = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, pressure_cfg, &zero_state);
  require(zero_probe.q_pred == 35 &&
              zero_probe.probe_issued,
          "a guarded censored zero margin permits only a shallow half-n release");

  pressure_cfg.q_cur = 35;
  const auto pressure_failure = hh::sizing::next_q_ss(
      efficiency_result, {moderate_error_snap}, pressure_cfg, &zero_state);
  require(pressure_failure.probe_failed &&
              pressure_failure.q_pred == 53 &&
              zero_state.difficulty.probe_retry_depth == 3,
          "a failed pressure probe halves the next admissible release depth");

  pressure_cfg.q_cur = 53;
  const auto pressure_recovery = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, pressure_cfg, &zero_state);
  require(pressure_recovery.q_pred == 53 &&
              !pressure_recovery.probe_issued &&
              zero_state.difficulty.guarded_demand == 0,
          "one sufficient recovery observation clears the failed-probe hold");
  const auto pressure_retry_wait = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, pressure_cfg, &zero_state);
  require(pressure_retry_wait.q_pred == 53 &&
              !pressure_retry_wait.probe_issued,
          "a recovered pressure probe again waits for guarded confirmation");
  const auto pressure_retry = hh::sizing::next_q_ss(
      efficiency_result, {efficiency_snap}, pressure_cfg, &zero_state);
  require(pressure_retry.q_pred == 50 &&
              pressure_retry.probe_issued,
          "the next pressure-authorized retry respects the halved failure depth");

  hh::GlobalResultLB resolved_result = efficiency_result;
  resolved_result.items = {
      {cheap_id, "cheap-ambiguity", 110, 0, 1, 1.0, 102, 120, false},
  };
  hh::sizing::PolicyState ambiguity_state;
  ambiguity_state.difficulty.ambiguity_margin = std::log(12.0 / 10.0);
  efficiency_cfg.residual_guard_decay = 0.5;
  const auto persisted = hh::sizing::next_q_ss(
      resolved_result, {efficiency_snap}, efficiency_cfg, &ambiguity_state);
  require(std::abs(persisted.b_q) < 1e-12,
          "ambiguity persistence does not alter the temporal residual bias");
  require(std::abs(persisted.g_amb - 0.5 * std::log(12.0 / 10.0)) < 1e-12 &&
              std::abs(ambiguity_state.difficulty.ambiguity_margin -
                       persisted.g_amb) < 1e-12,
          "unrefreshed ambiguity margin decays by the shared beta");

  cfg.kind = hh::sizing::PolicyKind::fixed;
  cfg.q_cur = 9;
  cfg.q_cap = 4;
  cfg.q_max = 5;
  const auto fixed = hh::sizing::next_q_ss(result, {s0, s1}, cfg, nullptr);
  require(fixed.q_next == 4, "fixed policy honors configured cap");
}

void test_hybrid_exact_head_and_tail_floor() {
  const Id128 hot = id_for("hot");
  const Id128 cold = id_for("cold");

  hh::HybridSS hybrid(1, 1);
  hybrid.seed_head({hot, cold});
  require(hybrid.head_size() == 1, "hybrid respects exact-head capacity while seeding");

  hybrid.update(hot, 5);
  hybrid.update(cold, 3);
  const auto snap = hybrid.snapshot_ex();
  require(snap.N_local == 8, "hybrid tracks combined local mass");
  require(snap.head_mass == 5, "hybrid exact head tracks seeded key exactly");
  require(snap.head_size == 1, "hybrid reports head size");

  const auto hot_item = std::find_if(snap.candidates.begin(), snap.candidates.end(),
                                     [&](const hh::Cand& c) { return c.id == hot; });
  require(hot_item != snap.candidates.end() && hot_item->est == 5,
          "hybrid snapshot includes exact head count");

  hybrid.reconfigure(1, 1, 10, 0.25);
  const auto resized = hybrid.snapshot_ex();
  require(resized.q_local >= 8, "hybrid tail capacity is floored by residual discoverability mass");
}

void test_json_gz_reader_orders_windows_and_partitions() {
  const auto root = std::filesystem::temp_directory_path() / "hh_tests_reader";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  write_gzip_json(root / "window_000002.json.gz", R"({"1":[["late",2]],"0":["early"]})");
  write_gzip_json(root / "window_000010.json.gz", R"({"0":["ten"]})");

  struct Event {
    std::size_t win;
    std::size_t part;
    std::string key;
    int weight;
  };
  std::vector<Event> events;
  const auto emitted = hh::JsonGzNestedReader::read(root.string(), [&](std::size_t w, std::size_t p,
                                                                       const std::string& k, int weight) {
    events.push_back({w, p, k, weight});
  });

  std::filesystem::remove_all(root);

  require(emitted == 3 && events.size() == 3, "reader emits expected event count");
  require(events[0].win == 2 && events[0].part == 0 && events[0].key == "early" && events[0].weight == 1,
          "reader infers window id from filename and orders numeric partitions");
  require(events[1].win == 2 && events[1].part == 1 && events[1].key == "late" && events[1].weight == 2,
          "reader preserves weighted entries");
  require(events[2].win == 10 && events[2].part == 0 && events[2].key == "ten",
          "reader processes directory files in lexical order while preserving parsed ids");
}

void run_all_tests() {
  init_hashing();
  test_space_saving_bounds_and_reconfigure();
  test_coordinator_certification_envelope();
  test_sizing_policy_outputs_are_clamped_and_finite();
  test_hybrid_exact_head_and_tail_floor();
  test_json_gz_reader_orders_windows_and_partitions();
}

} // namespace

int main() {
  try {
    run_all_tests();
  } catch (const TestFailure& e) {
    std::cerr << "TEST FAILURE: " << e.what() << '\n';
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "All hh tests passed\n";
  return EXIT_SUCCESS;
}
