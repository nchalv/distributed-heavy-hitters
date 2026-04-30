// apps/hh_run.cpp
// Interactive driver for a single sketch configuration. This mirrors the per-window evaluation
// described in the Adaptive Frequency Estimation paper: parse a nested JSON stream, simulate
// m partitions, reduce to a global view, and optionally resize SpaceSaving (or the hybrid tail)
// using the difficulty-aware controller from the paper.
#include "hh/io/reader.hpp"
#include "hh/core/hash.hpp"
#include "hh/core/arena_map.hpp"
#include "hh/sketches/isketch.hpp"
#include "hh/sketches/ss.hpp"
#include "hh/sketches/heavylocker.hpp"
#include "hh/sketches/chk.hpp"
#include "hh/oracle/oracle_all.hpp"
#include "hh/hybrid/hybrid.hpp"
#include "hh/coord/coordinator.hpp"
#include "hh/sizing/sizing.hpp"

#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace hh;

struct Partition {
  std::unique_ptr<ISketch> sketch;
  ArenaMap amap;
  std::unordered_map<Id128, std::string, Id128Hash> recent; // for admission-only binding
};

struct DifficultyVerification {
  bool valid{false};
  double tilde_qpred{NAN};
  std::size_t q_planned{0};
};

static void print_global_header(std::size_t win_idx, const std::string& method, std::size_t q_cur) {
  if (method == "ss") std::cout << "Window " << win_idx << " [ss q=" << q_cur << "]:\n";
  else std::cout << "Window " << win_idx << ":\n";
}

static void print_global(const GlobalResultLB& R) {
    std::cout << "  N_global=" << R.N_global
              << "  threshold(>1/n)=" << R.threshold
              << "  HH_count=" << R.items.size() << "\n";
    for (const auto& it : R.items) {
      if (it.cert_ub < R.threshold) continue; // certified non-HH
      const char* badge = it.guaranteed ? "[GUAR]" : "      ";
      std::string key = it.key.empty() ? Coordinator::id128_hex(it.id) : it.key;
      std::cout << "            " << badge << " [cert " << it.cert_lb << "," << it.cert_ub << "]"
                << " est=" << it.est
                << "  " << key << "\n";
    }
}

enum class Policy { Difficulty, Static };
enum class TailPolicy { Static, Difficulty };
enum class HeadPolicy { Candidate, TopN, Top2N, Threshold };
enum class DifficultyMode { Predictive, ReactiveReq, ReactiveEff };

struct Args {
  std::string path;
  std::size_t m{0};
  std::size_t n_param{0};
  std::size_t memKiB{0};
  std::string method;

  // adaptive flags
  Policy policy{Policy::Difficulty};
  double r{0.10};       // retained for compatibility with older inline specs
  double alpha_req{0.98}; // difficulty q_Req quantile
  // difficulty policy knobs
  double rho{0.0};
  double rho_a{0.5};
  std::size_t calib_window{3};
  double delta_m{0.2};
  std::size_t trend_h{3};
  double lambda_a{0.0};
  double lambda_g{0.0};
  double r_m{0.03};
  DifficultyMode diff_mode{DifficultyMode::Predictive};
  bool ss_per_item_eps{true};           // SS epsilon mode: false=global eps_max, true=per-item eps_i
  TailPolicy tail_policy{TailPolicy::Static}; // hybrid tail sizing
  std::size_t tail_q_factor{1};               // hybrid static tail: q_a = tail_q_factor * n_param
  HeadPolicy head_policy{HeadPolicy::Candidate}; // hybrid head seeding
  std::size_t hl_d{6};
  double hl_L{0.7};
  int hl_lossy{2};
  std::size_t hl_w{0};                        // optional fixed width (0 => auto from memory)
  std::size_t q_factor{1};                    // static SS: q = q_factor * n_param
};

static bool parse_args(int argc, char** argv, Args& a) {
  if (argc < 6) {
    std::cerr << "usage: " << argv[0]
              << " <stream.json.gz|stream_dir> <m> <n_param> <memKiB_each> <method:oracle|ss|hl|chk|hybrid>\n"
              << "       [--policy difficulty|static] [--alpha-req A]\n"
              << "       [--rho RHO] [--rho-a RA] [--calib-window L] [--delta-m D]\n"
              << "       [--trend-h H] [--lambda-a LA] [--lambda-g LG]\n"
              << "       [--r-m RM] [--diff-mode predictive|reactive-req|reactive-eff]\n"
              << "       [--ss-eps global|per-item]\n"
              << " [--hl-d D] [--hl-L L] [--hl-lossy MODE] [--hl-w W]\n"
              << "       [--hyb-tail n|2n|difficulty] [--hyb-head candidate|topn|top2n|threshold]\n"
              << "       [--q kN] (static SS: q = k * n_param, e.g., n, 2n)\n";
    return false;
  }
  a.path   = argv[1];
  a.m      = std::stoull(argv[2]);
  a.n_param= std::stoull(argv[3]);
  a.memKiB = std::stoull(argv[4]);
  a.method = argv[5];

  auto apply_kv = [&](const std::string& key, const std::string& val) -> bool {
    if (key == "policy") {
      if (val=="difficulty") a.policy=Policy::Difficulty;
      else if (val=="static") a.policy=Policy::Static;
      else { std::cerr<<"unknown policy "<<val<<"\n"; return false; }
    } else if (key == "r") {
      a.r = std::stod(val); if (!(a.r>0.0 && a.r<1.0)) a.r=0.10;
    } else if (key == "alpha-req") {
      a.alpha_req = std::stod(val); if (a.alpha_req<=0.0 || a.alpha_req>=1.0) a.alpha_req=0.98;
    } else if (key == "rho") {
      a.rho = std::clamp(std::stod(val), 0.0, 0.999999);
    } else if (key == "rho-a") {
      a.rho_a = std::clamp(std::stod(val), 0.0, 0.999999);
    } else if (key == "calib-window") {
      a.calib_window = std::max<std::size_t>(1, std::stoull(val));
    } else if (key == "delta-m") {
      a.delta_m = std::clamp(std::stod(val), 0.0, 1.0);
    } else if (key == "trend-h") {
      a.trend_h = std::max<std::size_t>(2, std::stoull(val));
    } else if (key == "lambda-a") {
      a.lambda_a = std::max(0.0, std::stod(val));
    } else if (key == "lambda-g") {
      a.lambda_g = std::max(0.0, std::stod(val));
    } else if (key == "r-m") {
      a.r_m = std::max(1e-12, std::stod(val));
    } else if (key == "diff-mode") {
      if (val == "predictive") a.diff_mode = DifficultyMode::Predictive;
      else if (val == "reactive-req") a.diff_mode = DifficultyMode::ReactiveReq;
      else if (val == "reactive-eff") a.diff_mode = DifficultyMode::ReactiveEff;
      else { std::cerr << "--diff-mode must be predictive|reactive-req|reactive-eff\n"; return false; }
    } else if (key == "ss-eps") {
      if (val=="global") a.ss_per_item_eps = false;
      else if (val=="per-item") a.ss_per_item_eps = true;
      else { std::cerr<<"--ss-eps must be global|per-item\n"; return false; }
    } else if (key == "hl-d") {
      a.hl_d = std::max<std::size_t>(1, std::stoull(val));
    } else if (key == "hl-L") {
      a.hl_L = std::stod(val);
    } else if (key == "hl-lossy") {
      a.hl_lossy = std::stoi(val);
    } else if (key == "hl-w") {
      a.hl_w = std::stoull(val);
    } else if (key == "hyb-tail") {
      std::string v = val;
      if (!v.empty() && v.back()=='n') {
        v.pop_back();
        a.tail_policy = TailPolicy::Static;
        a.tail_q_factor = v.empty() ? 1 : std::stoull(v);
        if (a.tail_q_factor == 0) a.tail_q_factor = 1;
      } else if (v=="difficulty") {
        a.tail_policy = TailPolicy::Difficulty;
      } else { std::cerr<<"--hyb-tail must be n|2n|difficulty\n"; return false; }
    } else if (key == "q") {
      std::string v = val;
      if (!v.empty() && v.back()=='n') v.pop_back();
      a.q_factor = v.empty() ? 1 : std::stoull(v);
      if (a.q_factor == 0) a.q_factor = 1;
    } else if (key == "hyb-head") {
      if      (val=="candidate") a.head_policy = HeadPolicy::Candidate;
      else if (val=="topn")      a.head_policy = HeadPolicy::TopN;
      else if (val=="top2n")     a.head_policy = HeadPolicy::Top2N;
      else if (val=="threshold") a.head_policy = HeadPolicy::Threshold;
      else { std::cerr<<"--hyb-head must be candidate|topn|top2n|threshold\n"; return false; }
    } else {
      std::cerr << "unknown option key '" << key << "'\n";
      return false;
    }
    return true;
  };

  // Support inline method spec:
  //   hybrid[hyb-tail=difficulty alpha-req=0.98 r-m=0.03]
  // including unquoted shell-split forms where options spill into following argv tokens.
  int i = 6;
  if (a.method.find('[') != std::string::npos) {
    std::string spec = a.method;
    while (spec.find(']') == std::string::npos && i < argc && std::string(argv[i]).rfind("--", 0) != 0) {
      spec += " ";
      spec += argv[i++];
    }
    const auto lb = spec.find('[');
    const auto rb = spec.rfind(']');
    if (lb == std::string::npos || rb == std::string::npos || rb < lb) {
      std::cerr << "Malformed method spec: " << spec << "\n";
      return false;
    }
    a.method = spec.substr(0, lb);
    const std::string opts = spec.substr(lb + 1, rb - lb - 1);
    std::stringstream ss(opts);
    std::string kv;
    while (ss >> kv) {
      const auto eq = kv.find('=');
      if (eq == std::string::npos) {
        std::cerr << "Malformed inline option '" << kv << "' in " << spec << "\n";
        return false;
      }
      const std::string key = kv.substr(0, eq);
      const std::string val = kv.substr(eq + 1);
      if (!apply_kv(key, val)) return false;
    }
  }

  for (; i<argc; ++i) {
    std::string flag = argv[i];
    auto need = [&](int more){ if (i+more >= argc){ std::cerr<<"missing arg for "<<flag<<"\n"; std::exit(2);} };
    if (flag == "--policy") {
      need(1); if (!apply_kv("policy", argv[++i])) return false;
    } else if (flag == "--r") {
      need(1); if (!apply_kv("r", argv[++i])) return false;
    } else if (flag == "--alpha-req") {
      need(1); if (!apply_kv("alpha-req", argv[++i])) return false;
    } else if (flag == "--rho") {
      need(1); if (!apply_kv("rho", argv[++i])) return false;
    } else if (flag == "--rho-a") {
      need(1); if (!apply_kv("rho-a", argv[++i])) return false;
    } else if (flag == "--calib-window") {
      need(1); if (!apply_kv("calib-window", argv[++i])) return false;
    } else if (flag == "--delta-m") {
      need(1); if (!apply_kv("delta-m", argv[++i])) return false;
    } else if (flag == "--trend-h") {
      need(1); if (!apply_kv("trend-h", argv[++i])) return false;
    } else if (flag == "--lambda-a") {
      need(1); if (!apply_kv("lambda-a", argv[++i])) return false;
    } else if (flag == "--lambda-g") {
      need(1); if (!apply_kv("lambda-g", argv[++i])) return false;
    } else if (flag == "--r-m") {
      need(1); if (!apply_kv("r-m", argv[++i])) return false;
    } else if (flag == "--diff-mode") {
      need(1); if (!apply_kv("diff-mode", argv[++i])) return false;
    } else if (flag == "--ss-eps") {
      need(1); if (!apply_kv("ss-eps", argv[++i])) return false;
    } else if (flag == "--hl-d") {
      need(1); if (!apply_kv("hl-d", argv[++i])) return false;
    } else if (flag == "--hl-L") {
      need(1); if (!apply_kv("hl-L", argv[++i])) return false;
    } else if (flag == "--hl-lossy") {
      need(1); if (!apply_kv("hl-lossy", argv[++i])) return false;
    } else if (flag == "--hl-w") {
      need(1); if (!apply_kv("hl-w", argv[++i])) return false;
    } else if (flag == "--hyb-tail") {
      need(1); if (!apply_kv("hyb-tail", argv[++i])) return false;
    } else if (flag == "--q") {
      need(1); if (!apply_kv("q", argv[++i])) return false;
    } else if (flag == "--hyb-head") {
      need(1); if (!apply_kv("hyb-head", argv[++i])) return false;
    }
  }
  if (a.m==0 || a.n_param==0) { std::cerr<<"m and n_param must be >0\n"; return false; }
  return true;
}

int main(int argc, char** argv) {
  Args A;
  if (!parse_args(argc, argv, A)) return 1;

  const std::size_t Mbytes = A.memKiB * 1024;
  set_secret_keys({0},{1},{2});

  std::vector<Partition> P(A.m);

  // SS: start q0 = n_param (subject to memory cap)
  std::size_t ss_q_cap = std::max<std::size_t>(std::size_t(1), Mbytes / 32); // ~32B per SS entry
  std::size_t ss_q_cur = 0;
  constexpr std::size_t HYB_HEAD_SLOT_BYTES = 24;
  constexpr std::size_t HYB_TAIL_SLOT_BYTES = 32;
  auto hybrid_tail_cap_for_head = [&](std::size_t q_e) -> std::size_t {
    const std::size_t head_bytes = q_e * HYB_HEAD_SLOT_BYTES;
    if (head_bytes >= Mbytes) return 1;
    return std::max<std::size_t>(std::size_t{1}, (Mbytes - head_bytes) / HYB_TAIL_SLOT_BYTES);
  };
  if (A.method == "ss") {
    const std::size_t desired_q = A.n_param * A.q_factor;
    ss_q_cur = std::min(desired_q, ss_q_cap); // first window: start at requested q (clamped to cap)
  }

  // Build partitions & sketches
  std::size_t hybrid_qe = 0, hybrid_qa = 0;
  if (A.method == "ss") {
    for (std::size_t i=0; i<A.m; ++i) {
      auto ss = std::make_unique<SpaceSaving>(ss_q_cur, A.ss_per_item_eps);
      ss->set_admission_callback([&, i](const Id128& id){
        auto it = P[i].recent.find(id);
        if (it == P[i].recent.end()) return;
        P[i].amap.bind_bytes(id, it->second);
      });
      P[i].sketch = std::move(ss);
    }
  } else if (A.method == "hl") {
    const std::size_t d = std::max<std::size_t>(1, A.hl_d);
    const double L = A.hl_L;
    const double theta = 1.0 / static_cast<double>(A.n_param);
    const int lossy_mode = A.hl_lossy;
    const double phi_occ = 0.90;
    const std::size_t SLOT_BYTES = 24;
    std::size_t w = A.hl_w;
    if (w == 0) {
      const std::size_t q_equiv = std::max<std::size_t>(1, (A.memKiB * 1024) / SLOT_BYTES);
      const double denom = static_cast<double>(d) * phi_occ;
      w = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(static_cast<double>(q_equiv) / denom)));
    }
    for (std::size_t i=0; i<A.m; ++i) {
      auto hl = std::make_unique<HeavyLocker>(w, d, L, theta, lossy_mode);
      hl->set_admission_callback([&, i](const Id128& id){
        auto it = P[i].recent.find(id);
        if (it == P[i].recent.end()) return;
        P[i].amap.bind_bytes(id, it->second);
      });
      P[i].sketch = std::move(hl);
    }
  } else if (A.method == "chk") {
    const std::size_t B = std::max<std::size_t>(1, Mbytes / 86); // ~86B/bucket across 2 tables
    for (std::size_t i=0; i<A.m; ++i) {
      auto chk = std::make_unique<CHK>(B, /*L*/16, /*decay*/1.08);
      chk->set_admission_callback([&, i](const Id128& id){
        auto it = P[i].recent.find(id);
        if (it == P[i].recent.end()) return;
        P[i].amap.bind_bytes(id, it->second);
      });
      P[i].sketch = std::move(chk);
    }
  } else if (A.method == "hybrid") {
    // First window: no prior head knowledge (q_e=0); tail starts at n_param
    hybrid_qe = 0;
    const std::size_t hybrid_tail_cap = hybrid_tail_cap_for_head(hybrid_qe);
    if (A.tail_policy == TailPolicy::Static) {
      hybrid_qa = A.n_param * std::max<std::size_t>(1, A.tail_q_factor);
    } else {
      hybrid_qa = A.n_param;
    }
    hybrid_qa = std::min<std::size_t>(hybrid_qa, hybrid_tail_cap);
    for (std::size_t i = 0; i < A.m; ++i) {
      auto hy = std::make_unique<HybridSS>(hybrid_qe, hybrid_qa);
      hy->set_admission_callback([&, i](const Id128& id){
        auto it = P[i].recent.find(id);
        if (it == P[i].recent.end()) return;
        P[i].amap.bind_bytes(id, it->second);
      });
      hy->reconfigure(hybrid_qe, hybrid_qa, A.n_param, /*head_mass_frac=*/0.0);
      P[i].sketch = std::move(hy);
    }
  } else if (A.method == "oracle") {
    for (std::size_t i=0; i<A.m; ++i) {
      auto ex = std::make_unique<OracleAll>();
      ex->set_admission_callback([&, i](const Id128& id){
        auto it = P[i].recent.find(id);
        if (it == P[i].recent.end()) return;
        P[i].amap.bind_bytes(id, it->second);
      });
      P[i].sketch = std::move(ex);
    }
  } else {
    std::cerr << "unknown method: " << A.method << "\n";
    return 3;
  }

  auto rebuild_ss_with_q = [&](std::size_t new_q){
    if (A.policy == Policy::Static) ss_q_cur = std::min<std::size_t>(A.n_param * A.q_factor, ss_q_cap);
    else ss_q_cur = std::max<std::size_t>(A.n_param * A.q_factor, new_q); // enforce q >= n
    for (std::size_t i=0; i<A.m; ++i) {
      auto ss = std::make_unique<SpaceSaving>(ss_q_cur, A.ss_per_item_eps);
      ss->set_admission_callback([&, i](const Id128& id){
        auto it = P[i].recent.find(id);
        if (it == P[i].recent.end()) return;
        P[i].amap.bind_bytes(id, it->second);
      });
      P[i].sketch = std::move(ss);
    }
  };

  std::size_t curr_win = 0; bool first_seen=false;
  // carryover for hybrid head seeding
  std::vector<Id128> hybrid_seed_ids;
  double hybrid_prev_head_frac = 0.0;
  sizing::PolicyState ss_policy_state{};
  sizing::PolicyState hybrid_tail_policy_state{};
  DifficultyVerification ss_verify{};
  DifficultyVerification hybrid_verify{};

  auto flush_window = [&](std::size_t win_idx){
    // End-of-window: reduce to global result, print telemetry, adapt sizes, then reset window state.
    std::vector<SnapshotEx> snaps; snaps.resize(A.m);
    std::vector<const ArenaMap*> maps; maps.resize(A.m);
    std::vector<HLBucketSnapshot> snaps_hl;
    bool all_hl_bucketed = (A.method == "hl");
    if (all_hl_bucketed) snaps_hl.reserve(A.m);
    for (std::size_t i=0; i<A.m; ++i) {
      snaps[i] = P[i].sketch->snapshot_ex();
      maps[i]  = &P[i].amap;
      if (all_hl_bucketed) {
        auto* hl = dynamic_cast<HeavyLocker*>(P[i].sketch.get());
        if (!hl) {
          all_hl_bucketed = false;
        } else {
          snaps_hl.push_back(hl->snapshot_bucketed());
        }
      }
    }

    GlobalResultLB R;
    if (all_hl_bucketed && snaps_hl.size() == A.m) {
      R = Coordinator::reduce_hl_bucketwise(snaps_hl, maps, A.n_param);
    } else {
      R = Coordinator::reduce_global_with_lb(snaps, maps, A.n_param);
    }
    print_global_header(win_idx, A.method, ss_q_cur);
    print_global(R);
    auto mem_kib_per_part = [&]() -> double {
      double sum_bytes = 0.0;
      std::size_t cnt = 0;
      for (const auto& part : P) {
        if (!part.sketch) continue;
        sum_bytes += static_cast<double>(part.sketch->memory_bytes());
        ++cnt;
      }
      return (cnt == 0) ? 0.0 : (sum_bytes / static_cast<double>(cnt)) / 1024.0;
    };
    if (A.method == "hybrid") {
      const double kib = mem_kib_per_part();
      std::cout << "  [hyb] actual_mem_per_part≈" << std::fixed << std::setprecision(1) << kib
                << " KiB (head=" << hybrid_qe << " tail=" << hybrid_qa << ")\n";
    } else if (A.method == "ss") {
      const double kib = mem_kib_per_part();
      std::cout << "  [ss] actual_mem_per_part≈" << std::fixed << std::setprecision(1) << kib
                << " KiB (q=" << ss_q_cur << ")\n";
    } else if (A.method == "hl") {
      const double kib = mem_kib_per_part();
      std::cout << "  [hl] actual_mem_per_part≈" << std::fixed << std::setprecision(1) << kib << " KiB\n";
    }

    if (A.method == "hybrid") {
      // Build candidate set C for head seeding and residual-tail sizing.
      hybrid_seed_ids.clear();
      std::unordered_set<Id128, Id128Hash> hybrid_candidate_ids;
      // compute min head mass fraction across partitions
      hybrid_prev_head_frac = 0.0;
      if (!snaps.empty()) {
        double min_frac = 1.0;
        for (const auto& s : snaps) {
          if (s.N_local == 0) { min_frac = 0.0; break; }
          const double frac = static_cast<double>(s.head_mass) / static_cast<double>(s.N_local);
          min_frac = std::min(min_frac, frac);
        }
        hybrid_prev_head_frac = min_frac;
      }
      std::size_t topk = std::min<std::size_t>(A.n_param, R.items.size());
      std::size_t top2k = std::min<std::size_t>(2 * A.n_param, R.items.size());

      for (std::size_t i = 0; i < R.items.size(); ++i) {
        const auto& it = R.items[i];
        const bool in_top = (i < topk);
        const bool in_top2 = (i < top2k);
        const bool challenger = (!in_top && it.cert_ub >= R.threshold);
        if (in_top || challenger) hybrid_candidate_ids.insert(it.id);
        if (A.head_policy == HeadPolicy::TopN) {
          if (in_top) hybrid_seed_ids.push_back(it.id);
        } else if (A.head_policy == HeadPolicy::Top2N) {
          if (in_top2) hybrid_seed_ids.push_back(it.id);
        } else if (A.head_policy == HeadPolicy::Threshold) {
          if (challenger) hybrid_seed_ids.push_back(it.id); // ub >= threshold
        } else {
          if (in_top || challenger) hybrid_seed_ids.push_back(it.id);
        }
      }
      hybrid_qe = std::max<std::size_t>(std::size_t{1}, hybrid_seed_ids.size()); // q_e = |C|
      const std::size_t hybrid_tail_cap = hybrid_tail_cap_for_head(hybrid_qe);
      // Tail sizing: approximate-only policy for q_a, with head coverage deduction
      const double P_E_min = std::clamp(hybrid_prev_head_frac, 0.0, 1.0); // min head coverage across partitions
      const std::size_t tail_floor = std::max<std::size_t>(
          std::size_t{1},
          static_cast<std::size_t>(std::ceil(static_cast<double>(A.n_param) * (1.0 - P_E_min))));
      if (A.tail_policy == TailPolicy::Static) {
        // Static tail: fix q_a to n (clamped by memory cap) rather than adapting with coverage
        const std::size_t q_static = A.n_param * std::max<std::size_t>(1, A.tail_q_factor);
        hybrid_qa = std::max<std::size_t>(tail_floor, std::min<std::size_t>(q_static, hybrid_tail_cap));
      } else {
        sizing::PolicyConfig cfg_h;
        cfg_h.n_param = A.n_param; // keep same n for policy math
        cfg_h.r = A.r;
        cfg_h.alpha_req = A.alpha_req;
        cfg_h.rho = A.rho;
        cfg_h.rho_a = A.rho_a;
        cfg_h.calib_window = A.calib_window;
        cfg_h.delta_m = A.delta_m;
        cfg_h.trend_h = A.trend_h;
        cfg_h.lambda_a = A.lambda_a;
        cfg_h.lambda_g = A.lambda_g;
        cfg_h.r_m = A.r_m;
        cfg_h.q_cur = hybrid_qa;
        cfg_h.q_min = tail_floor;
        cfg_h.q_cap = hybrid_tail_cap;
        cfg_h.q_max = hybrid_tail_cap;
        cfg_h.candidate_ids = &hybrid_candidate_ids;
        switch (A.tail_policy) {
          case TailPolicy::Difficulty: cfg_h.kind = sizing::PolicyKind::difficulty; break;
          case TailPolicy::Static: cfg_h.kind = sizing::PolicyKind::fixed; break;
        }
        auto res_h = sizing::next_q_ss(R, snaps, cfg_h, &hybrid_tail_policy_state);
        if (A.tail_policy == TailPolicy::Difficulty && hybrid_verify.valid) {
          const double residual_scale = std::max(0.0, 1.0 - P_E_min);
          const std::size_t q_req_resid = std::max<std::size_t>(
              tail_floor,
              static_cast<std::size_t>(std::ceil(static_cast<double>(res_h.q_req) * residual_scale)));
          const std::size_t q_eff_resid = std::max<std::size_t>(
              tail_floor,
              static_cast<std::size_t>(std::ceil(static_cast<double>(res_h.q_base) * residual_scale)));
          const std::size_t q_pred_eff_resid = std::max<std::size_t>(
              tail_floor,
              static_cast<std::size_t>(std::ceil(hybrid_verify.tilde_qpred * residual_scale)));
          const bool service_ok = (q_req_resid <= hybrid_verify.q_planned);
          const bool control_ok = (q_eff_resid <= hybrid_verify.q_planned);
          const bool calib_eff_ok = (q_eff_resid <= q_pred_eff_resid);
          std::cout << "  [hyb_tail verify] qReq_resid=" << q_req_resid
                    << " qEff_resid=" << q_eff_resid
                    << " | pred(qEff_resid<=" << q_pred_eff_resid
                    << ", q=" << hybrid_verify.q_planned << ")"
                    << " | service=" << (service_ok ? "OK" : "FAIL")
                    << " control=" << (control_ok ? "OK" : "FAIL")
                    << " calibEff=" << (calib_eff_ok ? "OK" : "FAIL") << "\n";
        }
        if (A.tail_policy == TailPolicy::Difficulty) {
          std::size_t q_next_mode = res_h.q_next;
          if (A.diff_mode == DifficultyMode::ReactiveReq) q_next_mode = std::max<std::size_t>(A.n_param, res_h.q_req);
          else if (A.diff_mode == DifficultyMode::ReactiveEff) q_next_mode = std::max<std::size_t>(A.n_param, res_h.q_base);
          if (cfg_h.q_cap != std::size_t(-1)) q_next_mode = std::min(q_next_mode, cfg_h.q_cap);
          q_next_mode = std::max(q_next_mode, cfg_h.q_min);
          if (cfg_h.q_max != std::size_t(-1)) q_next_mode = std::min(q_next_mode, cfg_h.q_max);
          res_h.q_next = q_next_mode;
          hybrid_verify.valid = true;
          hybrid_verify.tilde_qpred = res_h.q_pred_tilde;
        }
        // Paper: scale tail size by residual mass (1 - P_E^t).
        const double residual_scale = std::max(0.0, 1.0 - P_E_min);
        const std::size_t scaled_tail = static_cast<std::size_t>(std::ceil(res_h.q_next * residual_scale));
        hybrid_qa = std::max<std::size_t>(tail_floor, scaled_tail);
        if (A.tail_policy == TailPolicy::Difficulty) {
          // q_planned must match actual deployed residual tail.
          hybrid_verify.q_planned = hybrid_qa;
        }
        if (A.tail_policy == TailPolicy::Difficulty) {
          std::cout << "  [hyb_tail difficulty] Da=" << std::setprecision(4) << (std::isnan(res_h.da)?0.0:res_h.da)
                    << " alphaReq=" << A.alpha_req
                    << " bEff=" << std::setprecision(3) << (std::isnan(res_h.b_q)?0.0:res_h.b_q)
                    << " trend=" << std::setprecision(3) << (std::isnan(res_h.eta_a)?0.0:res_h.eta_a)
                    << " qEffPred=" << res_h.q_pred
                    << " qReq=" << res_h.q_req
                    << " qEffReplay=" << res_h.q_base << "\n";
        }
      }
      // reset state and seed head
      for (auto& part : P) {
        part.sketch->reset_window();
        part.amap.clear();
        part.recent.clear();
        auto* hy = dynamic_cast<HybridSS*>(part.sketch.get());
        if (hy) {
          hy->reconfigure(hybrid_qe, hybrid_qa, A.n_param, hybrid_prev_head_frac);
          hy->seed_head(hybrid_seed_ids);
        }
      }
      return;
    }

    if (A.method == "ss") {
      // Configure policy
      sizing::PolicyConfig cfg;
      // map policy enum
      switch (A.policy) {
        case Policy::Difficulty: cfg.kind = sizing::PolicyKind::difficulty; break;
        case Policy::Static: cfg.kind = sizing::PolicyKind::fixed; break;
      }
      cfg.n_param = A.n_param;
      cfg.r = A.r;
      cfg.alpha_req = A.alpha_req;
      cfg.rho = A.rho;
      cfg.rho_a = A.rho_a;
      cfg.calib_window = A.calib_window;
      cfg.delta_m = A.delta_m;
      cfg.trend_h = A.trend_h;
      cfg.lambda_a = A.lambda_a;
      cfg.lambda_g = A.lambda_g;
      cfg.r_m = A.r_m;
      cfg.q_cur = (A.policy == Policy::Static) ? (A.n_param * A.q_factor) : ss_q_cur;
      cfg.q_cap = ss_q_cap;
      cfg.q_min = (A.policy == Policy::Static) ? (A.n_param * A.q_factor) : A.n_param;
      cfg.q_max = ss_q_cap;
      std::unordered_set<Id128, Id128Hash> ss_candidate_ids;
      ss_candidate_ids.reserve(std::min<std::size_t>(R.items.size(), A.n_param * 2));
      const std::size_t topk = std::min<std::size_t>(A.n_param, R.items.size());
      for (std::size_t i = 0; i < R.items.size(); ++i) {
        const auto& it = R.items[i];
        const bool in_top = (i < topk);
        const bool challenger = (!in_top && it.cert_ub >= R.threshold);
        if (in_top || challenger) ss_candidate_ids.insert(it.id);
      }
      cfg.candidate_ids = &ss_candidate_ids;

      // Ask the shared policy engine
      auto pr = sizing::next_q_ss(R, snaps, cfg, &ss_policy_state);
      if (cfg.kind == sizing::PolicyKind::difficulty) {
        std::size_t q_next_mode = pr.q_next;
        if (A.diff_mode == DifficultyMode::ReactiveReq) q_next_mode = std::max<std::size_t>(A.n_param, pr.q_req);
        else if (A.diff_mode == DifficultyMode::ReactiveEff) q_next_mode = std::max<std::size_t>(A.n_param, pr.q_base);
        if (cfg.q_cap != std::size_t(-1)) q_next_mode = std::min(q_next_mode, cfg.q_cap);
        q_next_mode = std::max(q_next_mode, cfg.q_min);
        if (cfg.q_max != std::size_t(-1)) q_next_mode = std::min(q_next_mode, cfg.q_max);
        pr.q_next = q_next_mode;
      }
      if (cfg.kind == sizing::PolicyKind::difficulty && ss_verify.valid) {
        const bool service_ok = (pr.q_req <= ss_verify.q_planned);
        const bool control_ok = (pr.q_base <= ss_verify.q_planned);
        const bool calib_eff_ok = (pr.q_base <= static_cast<std::size_t>(std::llround(ss_verify.tilde_qpred)));
        std::cout << "  [ss verify] qReq=" << pr.q_req
                  << " qEff=" << pr.q_base
                  << " | pred(qEff<=" << static_cast<std::size_t>(std::llround(ss_verify.tilde_qpred))
                  << ", q=" << ss_verify.q_planned << ")"
                  << " | service=" << (service_ok ? "OK" : "FAIL")
                  << " control=" << (control_ok ? "OK" : "FAIL")
                  << " calibEff=" << (calib_eff_ok ? "OK" : "FAIL") << "\n";
      }
      if (cfg.kind == sizing::PolicyKind::difficulty) {
        ss_verify.valid = true;
        ss_verify.tilde_qpred = pr.q_pred_tilde;
        ss_verify.q_planned = pr.q_next;
      }

      // Print telemetry
      std::cout << std::fixed;
      std::cout << "  [ss] q_next=" << pr.q_next << " (policy=";
      if (cfg.kind == sizing::PolicyKind::difficulty) {
        std::cout << "difficulty, rho=" << A.rho
                  << ", rhoA=" << A.rho_a
                  << ", L=" << A.calib_window
                  << ", h=" << A.trend_h
                  << ", alphaReq=" << A.alpha_req
                  << ", lambdaA=" << A.lambda_a
                  << ", lambdaG=" << A.lambda_g
                  << ", mode="
                  << (A.diff_mode == DifficultyMode::Predictive ? "predictive"
                      : (A.diff_mode == DifficultyMode::ReactiveReq ? "reactive-req" : "reactive-eff"))
                  << ", Da=" << std::setprecision(4) << (std::isnan(pr.da)?0.0:pr.da)
                  << ", bEff=" << std::setprecision(3) << (std::isnan(pr.b_q)?0.0:pr.b_q)
                  << ", trend=" << std::setprecision(3) << (std::isnan(pr.eta_a)?0.0:pr.eta_a)
                  << ", qEffPred=" << pr.q_pred
                  << ", qReq=" << pr.q_req
                  << ", qEffReplay=" << pr.q_base;
      } else {
        std::cout << "fixed";
      }
      std::cout << ")\n";

      // Reset and rebuild SS with q_next
      for (auto& part : P){ part.sketch->reset_window(); part.amap.clear(); part.recent.clear(); }
      rebuild_ss_with_q((A.policy == Policy::Static) ? ss_q_cur : pr.q_next);
      return; // important: end-of-window handling for SS done
    }

    // Non-SS reset
    for (auto& part : P) {
      part.sketch->reset_window();
      part.amap.clear();
      part.recent.clear();
    }
  };

  // Ingestion
  JsonGzNestedReader::read(A.path, [&](std::size_t win, std::size_t part, const std::string& raw, int w){
    const std::size_t pidx = part % A.m;
    if (!first_seen) { curr_win = win; first_seen=true; }
    if (win != curr_win) { flush_window(curr_win); curr_win = win; }

    const Id128 id = id128_for(raw);

    // admission-only binding for all sketches (fairness policy)
    P[pidx].recent[id] = raw;
    P[pidx].sketch->update(id, w);
  });

  if (first_seen) flush_window(curr_win);
  return 0;
}
