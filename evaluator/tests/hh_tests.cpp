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
  cfg.q_max = 5;
  cfg.q_cap = 4;
  cfg.alpha_req = 1.0;
  cfg.r_m = 0.25;

  hh::sizing::PolicyState state;
  const auto first = hh::sizing::next_q_ss(result, {s0, s1}, cfg, &state);
  require(first.q_req >= cfg.n_param, "difficulty policy reports a requirement at least n");
  require(first.q_next >= cfg.q_min && first.q_next <= cfg.q_cap && first.q_next <= cfg.q_max,
          "difficulty policy clamps q_next");
  require(first.da >= 0.0 && first.da <= 1.0, "difficulty policy ambiguity mass is normalized");
  require(first.q_pred >= cfg.n_param, "difficulty policy exports predictive effective capacity");

  cfg.kind = hh::sizing::PolicyKind::fixed;
  cfg.q_cur = 9;
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
