// apps/hh_bench.cpp
// Batch benchmarking harness: replays the same nested JSON stream for multiple methods,
// treats the oracle pass as ground truth, and reports per-window plus aggregate accuracy
// metrics (HH precision/recall, AAE/ARE, optional top-k overlap) used in the paper’s plots.
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

#include "hh/bench/metrics.hpp"
#include "hh/sizing/sizing.hpp"   // centralized adaptive sizing

#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <optional>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>

using namespace hh;

struct Part {
  std::unique_ptr<ISketch> sketch;
  ArenaMap amap;
  std::unordered_map<Id128, std::string, Id128Hash> recent; // admission-only binding
};

struct MethodGroup {
  std::string name;
  std::string label; // unique for output (allows multiple of same method)
  std::vector<Part> parts; // size m
};

struct DifficultyVerification {
  bool valid{false};
  double tilde_qpred{NAN};
  std::size_t q_planned{0};
};

struct CertificationMetrics {
  std::size_t candidate_count{0};
  double candidate_hh_recall{0.0};
  std::size_t cert_pos_count{0};
  double cert_pos_precision{0.0};
  double cert_pos_mass{0.0};
  std::size_t ambiguous_count{0};
  double ambiguous_mass{0.0};
  std::size_t cert_neg_count{0};
  double cert_neg_mass{0.0};
  double interval_width_avg{0.0};
  double interval_width_amb_avg{0.0};
  double interval_width_max{0.0};
};

struct CsvRow {
  std::size_t window{0};
  std::string method;
  std::string method_type;
  std::uint64_t N_global{0};
  std::uint64_t threshold{0};

  double hh_precision{0.0};
  double hh_recall{0.0};
  double hh_f1{0.0};
  double aae{0.0};
  double are{0.0};
  bool has_topk{false};
  double topk_overlap{0.0};

  std::size_t q_current{0};
  std::size_t q_next{0};
  std::size_t q_head_current{0};
  std::size_t q_tail_current{0};
  std::size_t q_head_next{0};
  std::size_t q_tail_next{0};
  double margin_alpha{NAN};
  int service_violation{-1};
  std::size_t q_up{0};
  std::size_t q_baseline{0};
  int probe_issued{-1};
  int probe_failed{-1};
  std::size_t q_req_unclipped{0};
  std::size_t q_req{0};
  std::size_t q_eff_replay{0};
  std::size_t q_eff_pred{0};
  double q_eff_pred_tilde{NAN};
  int miss_req{-1};
  int miss_eff{-1};
  double over_req{NAN};
  double over_eff{NAN};
  double calib_bias{NAN};

  CertificationMetrics cert;

  double mem_worker_algo_kib{0.0};
  double mem_worker_key_kib{0.0};
  double mem_worker_total_kib{0.0};
  double mem_coord_input_kib{0.0};
  double mem_coord_work_kib{0.0};
  double mem_coord_peak_kib{0.0};

  std::uint64_t update_events{0};
  double update_ms{0.0};
  double update_mops{0.0};
  double reduce_ms{0.0};
  double control_ms{0.0};
};

static CertificationMetrics certification_metrics(const GlobalResultLB& oracle,
                                                   const GlobalResultLB& method) {
  CertificationMetrics cm{};
  cm.candidate_count = method.items.size();

  std::unordered_set<Id128, Id128Hash> oracle_hh;
  oracle_hh.reserve(oracle.items.size() * 2 + 8);
  for (const auto& it : oracle.items) {
    if (it.est >= oracle.threshold) oracle_hh.insert(it.id);
  }

  std::size_t candidate_tp = 0;
  std::size_t cert_pos_tp = 0;
  double width_sum = 0.0;
  double width_amb_sum = 0.0;
  const double N = static_cast<double>(method.N_global ? method.N_global : 1);

  for (const auto& it : method.items) {
    const bool is_oracle_hh = oracle_hh.find(it.id) != oracle_hh.end();
    if (is_oracle_hh) ++candidate_tp;

    const double mass = static_cast<double>(it.est) / N;
    const double width =
        static_cast<double>(it.cert_ub >= it.cert_lb ? it.cert_ub - it.cert_lb : 0) / N;
    width_sum += width;
    cm.interval_width_max = std::max(cm.interval_width_max, width);

    if (it.cert_lb >= method.threshold) {
      ++cm.cert_pos_count;
      cm.cert_pos_mass += mass;
      if (is_oracle_hh) ++cert_pos_tp;
    } else if (it.cert_ub < method.threshold) {
      ++cm.cert_neg_count;
      cm.cert_neg_mass += mass;
    } else {
      ++cm.ambiguous_count;
      cm.ambiguous_mass += mass;
      width_amb_sum += width;
    }
  }

  cm.candidate_hh_recall = oracle_hh.empty()
      ? 1.0
      : static_cast<double>(candidate_tp) / static_cast<double>(oracle_hh.size());
  cm.cert_pos_precision = cm.cert_pos_count == 0
      ? 1.0
      : static_cast<double>(cert_pos_tp) / static_cast<double>(cm.cert_pos_count);
  cm.interval_width_avg = cm.candidate_count == 0
      ? 0.0
      : width_sum / static_cast<double>(cm.candidate_count);
  cm.interval_width_amb_avg = cm.ambiguous_count == 0
      ? 0.0
      : width_amb_sum / static_cast<double>(cm.ambiguous_count);
  return cm;
}

static std::string csv_escape(const std::string& s) {
  bool quote = s.find_first_of(",\"\n\r") != std::string::npos;
  if (!quote) return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += "\"\"";
    else out.push_back(c);
  }
  out.push_back('"');
  return out;
}

static void write_csv_header(std::ostream& os) {
  os << "window,method,method_type,N_global,threshold,"
     << "hh_precision,hh_recall,hh_f1,aae,are,topk_overlap,"
     << "q_current,q_next,q_head_current,q_tail_current,q_head_next,q_tail_next,"
     << "margin_alpha,service_violation,q_up,q_baseline,"
     << "probe_issued,probe_failed,"
     << "q_req_unclipped,q_req,q_eff_replay,q_eff_pred,q_eff_pred_tilde,"
     << "miss_req,miss_eff,over_req,over_eff,"
     << "calib_bias,"
     << "candidate_count,candidate_hh_recall,cert_pos_count,cert_pos_precision,cert_pos_mass,"
     << "ambiguous_count,ambiguous_mass,cert_neg_count,cert_neg_mass,"
     << "interval_width_avg,interval_width_amb_avg,interval_width_max,"
     << "mem_worker_algo_kib,mem_worker_key_kib,mem_worker_total_kib,"
     << "mem_coord_input_kib,mem_coord_work_kib,mem_coord_peak_kib,"
     << "update_events,update_ms,update_mops,reduce_ms,control_ms\n";
}

static void write_csv_row(std::ostream& os, const CsvRow& r) {
  auto num = [](double v) {
    if (!std::isfinite(v)) return std::string{};
    std::ostringstream ss;
    ss << std::setprecision(10) << v;
    return ss.str();
  };
  auto opt_size = [](std::size_t v) {
    return v == 0 ? std::string{} : std::to_string(v);
  };
  auto opt_int = [](int v) {
    return v < 0 ? std::string{} : std::to_string(v);
  };

  os << r.window << ','
     << csv_escape(r.method) << ','
     << csv_escape(r.method_type) << ','
     << r.N_global << ','
     << r.threshold << ','
     << num(r.hh_precision) << ','
     << num(r.hh_recall) << ','
     << num(r.hh_f1) << ','
     << num(r.aae) << ','
     << num(r.are) << ','
     << (r.has_topk ? num(r.topk_overlap) : std::string{}) << ','
     << opt_size(r.q_current) << ','
     << opt_size(r.q_next) << ','
     << opt_size(r.q_head_current) << ','
     << opt_size(r.q_tail_current) << ','
     << opt_size(r.q_head_next) << ','
     << opt_size(r.q_tail_next) << ','
     << num(r.margin_alpha) << ','
     << opt_int(r.service_violation) << ','
     << opt_size(r.q_up) << ','
     << opt_size(r.q_baseline) << ','
     << opt_int(r.probe_issued) << ','
     << opt_int(r.probe_failed) << ','
     << r.q_req_unclipped << ','
     << opt_size(r.q_req) << ','
     << opt_size(r.q_eff_replay) << ','
     << opt_size(r.q_eff_pred) << ','
     << num(r.q_eff_pred_tilde) << ','
     << opt_int(r.miss_req) << ','
     << opt_int(r.miss_eff) << ','
     << num(r.over_req) << ','
     << num(r.over_eff) << ','
     << num(r.calib_bias) << ','
     << r.cert.candidate_count << ','
     << num(r.cert.candidate_hh_recall) << ','
     << r.cert.cert_pos_count << ','
     << num(r.cert.cert_pos_precision) << ','
     << num(r.cert.cert_pos_mass) << ','
     << r.cert.ambiguous_count << ','
     << num(r.cert.ambiguous_mass) << ','
     << r.cert.cert_neg_count << ','
     << num(r.cert.cert_neg_mass) << ','
     << num(r.cert.interval_width_avg) << ','
     << num(r.cert.interval_width_amb_avg) << ','
     << num(r.cert.interval_width_max) << ','
     << num(r.mem_worker_algo_kib) << ','
     << num(r.mem_worker_key_kib) << ','
     << num(r.mem_worker_total_kib) << ','
     << num(r.mem_coord_input_kib) << ','
     << num(r.mem_coord_work_kib) << ','
     << num(r.mem_coord_peak_kib) << ','
     << r.update_events << ','
     << num(r.update_ms) << ','
     << num(r.update_mops) << ','
     << num(r.reduce_ms) << ','
     << num(r.control_ms) << '\n';
}

// --- CLI policy/state ---
enum class Policy { Difficulty, Static };
enum class TailPolicy { Static, Difficulty };
enum class HeadPolicy { Candidate, TopN, Top2N, Threshold };
enum class DifficultyMode { Predictive, ReactiveReq, ReactiveEff };

struct Args {
  std::string path;
  std::size_t m{0};
  std::size_t n_param{0};
  std::size_t memKiB{0};
  std::string methods_csv;     // must include 'oracle' (reference)
  std::string csv_out;         // optional per-window CSV output
  std::optional<std::size_t> topk;

  // adaptive sizing knobs (for SS)
  Policy policy{Policy::Difficulty};
  double r{0.10};              // retained for compatibility with older method specs
  double alpha_req{0.98};      // difficulty: quantile for q_Req^t
  // difficulty policy knobs
  double delta_m{0.2};
  std::size_t res_guard_window{2};
  double res_guard_decay{0.5};
  bool symmetric_relaxation{false};
  bool censored_control{true};
  bool probe_residual_guard{true};
  hh::sizing::ProbeStrategy probe_strategy{hh::sizing::ProbeStrategy::bracket};
  double probe_pressure_gate{0.5};
  bool ambiguity_adjust{true};
  double r_m{0.03};
  DifficultyMode diff_mode{DifficultyMode::Predictive};
  bool ss_per_item_eps{true}; // SS epsilon mode: false=global eps_max, true=per-item eps_i

  // HeavyLocker parameters
  std::size_t hl_d{6};
  double hl_L{0.7};
  int hl_lossy{2}; // 0=MinusOne, 1=HeavyKeeper, 2=RAP, 3=USS
  std::size_t hl_w{0}; // optional fixed width (0 => auto from memory)

  TailPolicy tail_policy{TailPolicy::Static}; // hybrid tail sizing
  std::size_t tail_q_factor{1};               // hybrid static tail: q_a = tail_q_factor * n_param
  HeadPolicy head_policy{HeadPolicy::Candidate}; // hybrid head seeding
  std::size_t q_factor{1};                    // static SS: q = q_factor * n_param
};

struct MethodCfg {
  std::string name;
  Policy policy;
  double r;
  double alpha_req;
  double delta_m;
  std::size_t res_guard_window;
  double res_guard_decay;
  bool symmetric_relaxation;
  bool censored_control;
  bool probe_residual_guard;
  hh::sizing::ProbeStrategy probe_strategy;
  double probe_pressure_gate;
  bool ambiguity_adjust;
  double r_m;
  DifficultyMode diff_mode;
  bool ss_per_item_eps;
  std::size_t hl_d;
  double hl_L;
  int hl_lossy;
  std::size_t hl_w;
  TailPolicy tail_policy;
  std::size_t tail_q_factor;
  HeadPolicy head_policy;
  std::size_t q_factor;
};

static bool parse_args(int argc, char** argv, Args& a) {
  if (argc < 6) {
    std::cerr << "usage: " << argv[0]
              << " <stream.json.gz> <m> <n_param> <memKiB_each> <methods_csv>"
              << " [--csv-out FILE] [--topk K] [--policy difficulty|static] [--alpha-req A]"
              << " [--delta-m D (deprecated, ignored)]"
              << " [--res-guard-window W] [--res-guard-decay B]"
              << " [--symmetric-relaxation on|off]"
              << " [--downward-probing on|off]"
              << " [--probe-residual-guard on|off]"
              << " [--probe-strategy bracket|comfort|pressure]"
              << " [--probe-pressure-gate RHO]"
              << " [--amb-adjust on|off]"
              << " [--r-m RM] [--diff-mode predictive|reactive-req|reactive-eff]"
              << " [--ss-eps global|per-item]"
              << " [--hl-d D] [--hl-L L] [--hl-lossy MODE] [--hl-w W]"
              << " [--hyb-tail n|2n|difficulty] [--hyb-head candidate|topn|top2n|threshold] [--q kN]\n";
    return false;
  }
  a.path       = argv[1];
  a.m          = std::stoull(argv[2]);
  a.n_param    = std::stoull(argv[3]);
  a.memKiB     = std::stoull(argv[4]);
  a.methods_csv= argv[5];

  for (int i=6; i<argc; ++i) {
    std::string flag = argv[i];
  auto need = [&](int more){ if (i+more >= argc){ std::cerr<<"missing arg for "<<flag<<"\n"; std::exit(2);} };
  if (flag == "--csv-out") { need(1); a.csv_out = argv[++i]; }
  else if (flag == "--topk") { need(1); a.topk = std::stoull(argv[++i]); }
  else if (flag == "--policy") {
    need(1); std::string v = argv[++i];
    if (v=="difficulty") a.policy = Policy::Difficulty;
    else if (v=="static") a.policy = Policy::Static;
    else { std::cerr<<"unknown policy "<<v<<"\n"; return false; }
  } else if (flag == "--alpha-req") {
    need(1); a.alpha_req = std::stod(argv[++i]); if (a.alpha_req<=0 || a.alpha_req>=1) a.alpha_req=0.98;
  } else if (flag == "--delta-m") {
    need(1); a.delta_m = std::clamp(std::stod(argv[++i]), 0.0, 1.0);
  } else if (flag == "--res-guard-window") {
    need(1); a.res_guard_window = std::stoull(argv[++i]);
  } else if (flag == "--res-guard-decay") {
    need(1); a.res_guard_decay = std::clamp(std::stod(argv[++i]), 0.0, 1.0);
  } else if (flag == "--symmetric-relaxation") {
    need(1);
    std::string v = argv[++i];
    if (v == "on" || v == "true" || v == "1") a.symmetric_relaxation = true;
    else if (v == "off" || v == "false" || v == "0") a.symmetric_relaxation = false;
    else { std::cerr<<"--symmetric-relaxation must be on|off\n"; return false; }
  } else if (flag == "--censored-control" || flag == "--downward-probing") {
    need(1);
    std::string v = argv[++i];
    if (v == "on" || v == "true" || v == "1") a.censored_control = true;
    else if (v == "off" || v == "false" || v == "0") a.censored_control = false;
    else { std::cerr<<"--downward-probing must be on|off\n"; return false; }
  } else if (flag == "--probe-residual-guard") {
    need(1);
    std::string v = argv[++i];
    if (v == "on" || v == "true" || v == "1") a.probe_residual_guard = true;
    else if (v == "off" || v == "false" || v == "0") a.probe_residual_guard = false;
    else { std::cerr<<"--probe-residual-guard must be on|off\n"; return false; }
  } else if (flag == "--probe-strategy") {
    need(1);
    std::string v = argv[++i];
    if (v == "bracket") a.probe_strategy = hh::sizing::ProbeStrategy::bracket;
    else if (v == "comfort") a.probe_strategy = hh::sizing::ProbeStrategy::comfort;
    else if (v == "pressure") a.probe_strategy = hh::sizing::ProbeStrategy::pressure;
    else { std::cerr<<"--probe-strategy must be bracket|comfort|pressure\n"; return false; }
  } else if (flag == "--probe-pressure-gate") {
    need(1); a.probe_pressure_gate = std::clamp(std::stod(argv[++i]), 0.0, 1.0);
  } else if (flag == "--amb-adjust") {
    need(1);
    std::string v = argv[++i];
    if (v == "on" || v == "true" || v == "1") a.ambiguity_adjust = true;
    else if (v == "off" || v == "false" || v == "0") a.ambiguity_adjust = false;
    else { std::cerr<<"--amb-adjust must be on|off\n"; return false; }
  } else if (flag == "--r-m") {
    need(1); a.r_m = std::max(1e-12, std::stod(argv[++i]));
  } else if (flag == "--diff-mode") {
    need(1);
    std::string v = argv[++i];
    if (v == "predictive") a.diff_mode = DifficultyMode::Predictive;
    else if (v == "reactive-req") a.diff_mode = DifficultyMode::ReactiveReq;
    else if (v == "reactive-eff") a.diff_mode = DifficultyMode::ReactiveEff;
    else { std::cerr<<"--diff-mode must be predictive|reactive-req|reactive-eff\n"; return false; }
  } else if (flag == "--ss-eps") {
    need(1); std::string v = argv[++i];
    if      (v=="global") a.ss_per_item_eps = false;
    else if (v=="per-item") a.ss_per_item_eps = true;
    else { std::cerr<<"--ss-eps must be global|per-item\n"; return false; }
  } else if (flag == "--hl-d") {
    need(1); a.hl_d = std::max<std::size_t>(1, std::stoull(argv[++i]));
  } else if (flag == "--hl-L") {
    need(1); a.hl_L = std::stod(argv[++i]);
  } else if (flag == "--hl-lossy") {
    need(1); a.hl_lossy = std::stoi(argv[++i]);
  } else if (flag == "--hl-w") {
    need(1); a.hl_w = std::stoull(argv[++i]);
  } else if (flag == "--hyb-tail") {
    need(1); std::string v = argv[++i];
    if (!v.empty() && v.back()=='n') {
      v.pop_back();
      a.tail_policy = TailPolicy::Static;
      a.tail_q_factor = v.empty() ? 1 : std::stoull(v);
      if (a.tail_q_factor == 0) a.tail_q_factor = 1;
    } else if (v=="difficulty") {
      a.tail_policy = TailPolicy::Difficulty;
    } else { std::cerr<<"--hyb-tail must be n|2n|difficulty\n"; return false; }
  } else if (flag == "--q") {
    need(1); std::string v = argv[++i];
    if (!v.empty() && v.back()=='n') v.pop_back();
    a.q_factor = v.empty() ? 1 : std::stoull(v);
    if (a.q_factor == 0) a.q_factor = 1;
  } else if (flag == "--hyb-head") {
    need(1); std::string v = argv[++i];
    if      (v=="candidate") a.head_policy = HeadPolicy::Candidate;
    else if (v=="topn")      a.head_policy = HeadPolicy::TopN;
    else if (v=="top2n")     a.head_policy = HeadPolicy::Top2N;
    else if (v=="threshold") a.head_policy = HeadPolicy::Threshold;
    else { std::cerr<<"--hyb-head must be candidate|topn|top2n|threshold\n"; return false; }
  }
}

  if (a.m==0 || a.n_param==0) { std::cerr<<"m and n_param must be >0\n"; return false; }
  return true;
}

// --- method builders ---
static MethodGroup make_method(const MethodCfg& spec, std::size_t m, std::size_t bytes_per_part, std::size_t n_param) {
  MethodGroup G; G.name = spec.name; G.parts.resize(m);

  if (spec.name == "oracle") {
    for (std::size_t i=0; i<m; ++i) {
      auto ex = std::make_unique<OracleAll>();
      Part* part = &G.parts[i];
      ex->set_admission_callback([part](const Id128& id){
        auto it = part->recent.find(id);
        if (it == part->recent.end()) return;
        part->amap.bind_bytes(id, it->second);
      });
      G.parts[i].sketch = std::move(ex);
    }
  } else if (spec.name == "ss") {
    // initial q0 = n (cap to memory): ~32B per SS counter
    const std::size_t q_cap = std::max<std::size_t>(std::size_t(1), bytes_per_part / 32);
    const std::size_t q0 = std::min<std::size_t>(n_param, q_cap); // initial guess; will be set per policy
    for (std::size_t i=0; i<m; ++i) {
      auto ss = std::make_unique<SpaceSaving>(q0, spec.ss_per_item_eps);
      Part* part = &G.parts[i];
      ss->set_admission_callback([part](const Id128& id){
        auto it = part->recent.find(id);
        if (it == part->recent.end()) return;
        part->amap.bind_bytes(id, it->second);
      });
      G.parts[i].sketch = std::move(ss);
    }
  } else if (spec.name == "hl") {
    // HeavyLocker (paper-inspired static sizing)
    const std::size_t d = std::max<std::size_t>(1, spec.hl_d);
    const double L = spec.hl_L;
    const double theta = 1.0 / static_cast<double>(n_param); // θ = 1/n
    const int lossy_mode = spec.hl_lossy;
    const double phi_occ = 0.90;
    const std::size_t SLOT_BYTES = 24;                       // ≈16(id)+4(cnt)+pad
    std::size_t w = spec.hl_w;
    if (w == 0) {
      const std::size_t q_equiv = std::max<std::size_t>(1, bytes_per_part / SLOT_BYTES);
      const double denom = static_cast<double>(d) * phi_occ;
      w = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(static_cast<double>(q_equiv) / denom)));
    }
    for (auto& p : G.parts) {
      auto hl = std::make_unique<HeavyLocker>(w, d, L, theta, lossy_mode);
      Part* part = &p;
      hl->set_admission_callback([part](const Id128& id){
        auto it = part->recent.find(id);
        if (it == part->recent.end()) return;
        part->amap.bind_bytes(id, it->second);
      });
      p.sketch = std::move(hl);
    }
  } else if (spec.name == "chk") {
    // CHK: ~86B per bucket across 2 tables (empirical)
    const std::size_t B = std::max<std::size_t>(1, bytes_per_part / 86);
    for (auto& p : G.parts) {
      auto chk = std::make_unique<CHK>(B, /*L*/16, /*decay*/1.08);
      Part* part = &p;
      chk->set_admission_callback([part](const Id128& id){
        auto it = part->recent.find(id);
        if (it == part->recent.end()) return;
        part->amap.bind_bytes(id, it->second);
      });
      p.sketch = std::move(chk);
    }
  } else if (spec.name == "hybrid") {
    // Rough split: half memory to head (24B/entry), half to tail (32B/entry)
    const std::size_t head_bytes = bytes_per_part / 2;
    const std::size_t tail_bytes = bytes_per_part - head_bytes;
    const std::size_t q_e = std::max<std::size_t>(1, head_bytes / 24);
    const std::size_t q_a = std::max<std::size_t>(n_param, tail_bytes / 32); // residual discoverability
    for (auto& p : G.parts) {
      auto hy = std::make_unique<HybridSS>(q_e, q_a);
      Part* part = &p;
      hy->set_admission_callback([part](const Id128& id){
        auto it = part->recent.find(id);
        if (it == part->recent.end()) return;
        part->amap.bind_bytes(id, it->second);
      });
      p.sketch = std::move(hy);
    }
  }

  return G;
}

static GlobalResultLB reduce_global(const MethodGroup& G, std::size_t n_param) {
  std::vector<SnapshotEx> snaps; snaps.reserve(G.parts.size());
  std::vector<const ArenaMap*> maps; maps.reserve(G.parts.size());
  for (const auto& p : G.parts) {
    snaps.push_back(p.sketch->snapshot_ex());
    maps.push_back(&p.amap);
  }
  return Coordinator::reduce_global_with_lb(snaps, maps, n_param);
}

// helper: rebuild ss group with a new q across partitions
static void rebuild_ss_group(MethodGroup& G, std::size_t q_new, bool per_item_eps) {
  for (auto& p : G.parts) {
    auto ss = std::make_unique<SpaceSaving>(q_new, per_item_eps);
    Part* part = &p;
    ss->set_admission_callback([part](const Id128& id){
      auto it = part->recent.find(id);
      if (it == part->recent.end()) return;
      part->amap.bind_bytes(id, it->second);
    });
    p.sketch = std::move(ss);
  }
}

int main(int argc, char** argv) {
  Args A;
  if (!parse_args(argc, argv, A)) return 1;

  std::ofstream csv_file;
  if (!A.csv_out.empty()) {
    csv_file.open(A.csv_out);
    if (!csv_file) {
      std::cerr << "ERROR: could not open CSV output file '" << A.csv_out << "'\n";
      return 5;
    }
    write_csv_header(csv_file);
  }

  set_secret_keys({0},{1},{2});
  const std::size_t bytes_per_part = A.memKiB * 1024;

  // Parse method list
  auto known_method = [&](const std::string& nm){
    return nm == "oracle" || nm == "hl" || nm == "chk" || nm == "hybrid" || nm == "ss";
  };
  std::vector<MethodCfg> methods;
  {
    MethodCfg defaults{"", A.policy, A.r, A.alpha_req,
                      A.delta_m, A.res_guard_window, A.res_guard_decay,
                      A.symmetric_relaxation,
                      A.censored_control,
                      A.probe_residual_guard,
                      A.probe_strategy,
                      A.probe_pressure_gate,
                      A.ambiguity_adjust,
                      A.r_m, A.diff_mode, A.ss_per_item_eps,
                      A.hl_d, A.hl_L, A.hl_lossy, A.hl_w,
                      A.tail_policy, A.tail_q_factor, A.head_policy, A.q_factor};
    std::string tok;
    bool in_bracket = false;
    for (char c : A.methods_csv) {
      if (c == ',' && !in_bracket) {
        if (!tok.empty()) { // flush token
          // parse token
          std::string name = tok;
          std::string opts;
          auto lb = tok.find('[');
          if (lb != std::string::npos) {
            name = tok.substr(0, lb);
            auto rb = tok.find(']', lb);
            if (rb == std::string::npos || rb != tok.size()-1) {
              std::cerr << "Malformed method entry: " << tok << "\n";
              return 4;
            }
            opts = tok.substr(lb+1, rb-lb-1);
          }
          if (!known_method(name)) {
            std::cerr << "ERROR: unknown method '" << name << "'. Allowed: oracle, ss, hl, chk, hybrid.\n";
            return 4;
          }
          MethodCfg cfg = defaults;
          cfg.name = name;
          if (!opts.empty()) {
            std::stringstream ss(opts);
            std::string kv;
            while (ss >> kv) {
              auto eq = kv.find('=');
              if (eq == std::string::npos) { std::cerr << "Malformed option '"<<kv<<"' in " << tok << "\n"; return 4; }
              auto key = kv.substr(0, eq);
              auto val = kv.substr(eq+1);
              if (key == "policy") {
                if (val=="difficulty") cfg.policy = Policy::Difficulty;
                else if (val=="static") cfg.policy = Policy::Static;
                else { std::cerr<<"unknown policy "<<val<<"\n"; return 4; }
              } else if (key == "r") {
                cfg.r = std::stod(val); if (!(cfg.r>0 && cfg.r<1)) cfg.r=defaults.r;
              } else if (key == "alpha-req") {
                cfg.alpha_req = std::stod(val); if (cfg.alpha_req<=0 || cfg.alpha_req>=1) cfg.alpha_req=defaults.alpha_req;
              } else if (key == "delta-m") {
                cfg.delta_m = std::clamp(std::stod(val), 0.0, 1.0);
              } else if (key == "res-guard-window") {
                cfg.res_guard_window = std::stoull(val);
              } else if (key == "res-guard-decay") {
                cfg.res_guard_decay = std::clamp(std::stod(val), 0.0, 1.0);
              } else if (key == "symmetric-relaxation") {
                if (val == "on" || val == "true" || val == "1") cfg.symmetric_relaxation = true;
                else if (val == "off" || val == "false" || val == "0") cfg.symmetric_relaxation = false;
                else { std::cerr<<"symmetric-relaxation must be on|off\n"; return 4; }
              } else if (key == "censored-control" || key == "downward-probing") {
                if (val == "on" || val == "true" || val == "1") cfg.censored_control = true;
                else if (val == "off" || val == "false" || val == "0") cfg.censored_control = false;
                else { std::cerr<<"downward-probing must be on|off\n"; return 4; }
              } else if (key == "probe-residual-guard") {
                if (val == "on" || val == "true" || val == "1") cfg.probe_residual_guard = true;
                else if (val == "off" || val == "false" || val == "0") cfg.probe_residual_guard = false;
                else { std::cerr<<"probe-residual-guard must be on|off\n"; return 4; }
              } else if (key == "probe-strategy") {
                if (val == "bracket") cfg.probe_strategy = hh::sizing::ProbeStrategy::bracket;
                else if (val == "comfort") cfg.probe_strategy = hh::sizing::ProbeStrategy::comfort;
                else if (val == "pressure") cfg.probe_strategy = hh::sizing::ProbeStrategy::pressure;
                else { std::cerr<<"probe-strategy must be bracket|comfort|pressure\n"; return 4; }
              } else if (key == "probe-pressure-gate") {
                cfg.probe_pressure_gate = std::clamp(std::stod(val), 0.0, 1.0);
              } else if (key == "amb-adjust") {
                if (val == "on" || val == "true" || val == "1") cfg.ambiguity_adjust = true;
                else if (val == "off" || val == "false" || val == "0") cfg.ambiguity_adjust = false;
                else { std::cerr<<"amb-adjust must be on|off\n"; return 4; }
              } else if (key == "r-m") {
                cfg.r_m = std::max(1e-12, std::stod(val));
              } else if (key == "diff-mode") {
                if (val == "predictive") cfg.diff_mode = DifficultyMode::Predictive;
                else if (val == "reactive-req") cfg.diff_mode = DifficultyMode::ReactiveReq;
                else if (val == "reactive-eff") cfg.diff_mode = DifficultyMode::ReactiveEff;
                else { std::cerr<<"diff-mode must be predictive|reactive-req|reactive-eff\n"; return 4; }
              } else if (key == "ss-eps") {
                if      (val=="global") cfg.ss_per_item_eps = false;
                else if (val=="per-item") cfg.ss_per_item_eps = true;
                else { std::cerr<<"ss-eps must be global|per-item\n"; return 4; }
              } else if (key == "hyb-tail") {
                if (!val.empty() && val.back()=='n') {
                  val.pop_back();
                  cfg.tail_policy = TailPolicy::Static;
                  cfg.tail_q_factor = val.empty() ? 1 : std::stoull(val);
                  if (cfg.tail_q_factor == 0) cfg.tail_q_factor = 1;
                } else if (val=="difficulty") {
                  cfg.tail_policy = TailPolicy::Difficulty;
                } else { std::cerr<<"hyb-tail must be n|2n|difficulty\n"; return 4; }
              } else if (key == "q") {
                std::string v = val;
                if (!v.empty() && v.back()=='n') v.pop_back();
                cfg.q_factor = v.empty() ? 1 : std::stoull(v);
                if (cfg.q_factor == 0) cfg.q_factor = 1;
              } else if (key == "hyb-head") {
                if      (val=="candidate") cfg.head_policy = HeadPolicy::Candidate;
                else if (val=="topn")      cfg.head_policy = HeadPolicy::TopN;
                else if (val=="top2n")     cfg.head_policy = HeadPolicy::Top2N;
                else if (val=="threshold") cfg.head_policy = HeadPolicy::Threshold;
                else { std::cerr<<"hyb-head must be candidate|topn|top2n|threshold\n"; return 4; }
              } else if (key == "hl-d") {
                cfg.hl_d = std::max<std::size_t>(1, std::stoull(val));
              } else if (key == "hl-L") {
                cfg.hl_L = std::stod(val);
              } else if (key == "hl-lossy") {
                cfg.hl_lossy = std::stoi(val);
              } else if (key == "hl-w") {
                cfg.hl_w = std::stoull(val);
              } else {
                std::cerr << "Unknown method option '" << key << "'\n";
                return 4;
              }
            }
          }
          methods.push_back(cfg);
          tok.clear();
        }
      } else {
        if (c == '[') in_bracket = true;
        if (c == ']') in_bracket = false;
        tok.push_back(c);
      }
    }
  if (!tok.empty()) {
      std::string name = tok;
      std::string opts;
      auto lb = tok.find('[');
      if (lb != std::string::npos) {
        name = tok.substr(0, lb);
        auto rb = tok.find(']', lb);
        if (rb == std::string::npos || rb != tok.size()-1) {
          std::cerr << "Malformed method entry: " << tok << "\n";
          return 4;
        }
        opts = tok.substr(lb+1, rb-lb-1);
      }
      if (!known_method(name)) {
        std::cerr << "ERROR: unknown method '" << name << "'. Allowed: oracle, ss, hl, chk, hybrid.\n";
        return 4;
      }
      MethodCfg cfg = defaults;
      cfg.name = name;
      if (!opts.empty()) {
        std::stringstream ss(opts);
        std::string kv;
        while (ss >> kv) {
          auto eq = kv.find('=');
          if (eq == std::string::npos) { std::cerr << "Malformed option '"<<kv<<"' in " << tok << "\n"; return 4; }
          auto key = kv.substr(0, eq);
          auto val = kv.substr(eq+1);
          if (key == "policy") {
            if (val=="difficulty") cfg.policy = Policy::Difficulty;
            else if (val=="static") cfg.policy = Policy::Static;
            else { std::cerr<<"unknown policy "<<val<<"\n"; return 4; }
          } else if (key == "r") {
            cfg.r = std::stod(val); if (!(cfg.r>0 && cfg.r<1)) cfg.r=defaults.r;
          } else if (key == "alpha-req") {
            cfg.alpha_req = std::stod(val); if (cfg.alpha_req<=0 || cfg.alpha_req>=1) cfg.alpha_req=defaults.alpha_req;
          } else if (key == "delta-m") {
            cfg.delta_m = std::clamp(std::stod(val), 0.0, 1.0);
          } else if (key == "res-guard-window") {
            cfg.res_guard_window = std::stoull(val);
          } else if (key == "res-guard-decay") {
            cfg.res_guard_decay = std::clamp(std::stod(val), 0.0, 1.0);
          } else if (key == "symmetric-relaxation") {
            if (val == "on" || val == "true" || val == "1") cfg.symmetric_relaxation = true;
            else if (val == "off" || val == "false" || val == "0") cfg.symmetric_relaxation = false;
            else { std::cerr<<"symmetric-relaxation must be on|off\n"; return 4; }
          } else if (key == "censored-control" || key == "downward-probing") {
            if (val == "on" || val == "true" || val == "1") cfg.censored_control = true;
            else if (val == "off" || val == "false" || val == "0") cfg.censored_control = false;
            else { std::cerr<<"downward-probing must be on|off\n"; return 4; }
          } else if (key == "probe-residual-guard") {
            if (val == "on" || val == "true" || val == "1") cfg.probe_residual_guard = true;
            else if (val == "off" || val == "false" || val == "0") cfg.probe_residual_guard = false;
            else { std::cerr<<"probe-residual-guard must be on|off\n"; return 4; }
          } else if (key == "probe-strategy") {
            if (val == "bracket") cfg.probe_strategy = hh::sizing::ProbeStrategy::bracket;
            else if (val == "comfort") cfg.probe_strategy = hh::sizing::ProbeStrategy::comfort;
            else if (val == "pressure") cfg.probe_strategy = hh::sizing::ProbeStrategy::pressure;
            else { std::cerr<<"probe-strategy must be bracket|comfort|pressure\n"; return 4; }
          } else if (key == "probe-pressure-gate") {
            cfg.probe_pressure_gate = std::clamp(std::stod(val), 0.0, 1.0);
          } else if (key == "amb-adjust") {
            if (val == "on" || val == "true" || val == "1") cfg.ambiguity_adjust = true;
            else if (val == "off" || val == "false" || val == "0") cfg.ambiguity_adjust = false;
            else { std::cerr<<"amb-adjust must be on|off\n"; return 4; }
          } else if (key == "r-m") {
            cfg.r_m = std::max(1e-12, std::stod(val));
          } else if (key == "diff-mode") {
            if (val == "predictive") cfg.diff_mode = DifficultyMode::Predictive;
            else if (val == "reactive-req") cfg.diff_mode = DifficultyMode::ReactiveReq;
            else if (val == "reactive-eff") cfg.diff_mode = DifficultyMode::ReactiveEff;
            else { std::cerr<<"diff-mode must be predictive|reactive-req|reactive-eff\n"; return 4; }
          } else if (key == "ss-eps") {
            if      (val=="global") cfg.ss_per_item_eps = false;
            else if (val=="per-item") cfg.ss_per_item_eps = true;
            else { std::cerr<<"ss-eps must be global|per-item\n"; return 4; }
          } else if (key == "hyb-tail") {
            if (!val.empty() && val.back()=='n') {
              val.pop_back();
              cfg.tail_policy = TailPolicy::Static;
              cfg.tail_q_factor = val.empty() ? 1 : std::stoull(val);
              if (cfg.tail_q_factor == 0) cfg.tail_q_factor = 1;
            } else if (val=="difficulty") {
              cfg.tail_policy = TailPolicy::Difficulty;
            } else { std::cerr<<"hyb-tail must be n|2n|difficulty\n"; return 4; }
          } else if (key == "q") {
            std::string v = val;
            if (!v.empty() && v.back()=='n') v.pop_back();
            cfg.q_factor = v.empty() ? 1 : std::stoull(v);
            if (cfg.q_factor == 0) cfg.q_factor = 1;
          } else if (key == "hyb-head") {
            if      (val=="candidate") cfg.head_policy = HeadPolicy::Candidate;
            else if (val=="topn")      cfg.head_policy = HeadPolicy::TopN;
            else if (val=="top2n")     cfg.head_policy = HeadPolicy::Top2N;
            else if (val=="threshold") cfg.head_policy = HeadPolicy::Threshold;
            else { std::cerr<<"hyb-head must be candidate|topn|top2n|threshold\n"; return 4; }
          } else if (key == "hl-d") {
            cfg.hl_d = std::max<std::size_t>(1, std::stoull(val));
          } else if (key == "hl-L") {
            cfg.hl_L = std::stod(val);
          } else if (key == "hl-lossy") {
            cfg.hl_lossy = std::stoi(val);
          } else if (key == "hl-w") {
            cfg.hl_w = std::stoull(val);
          } else {
            std::cerr << "Unknown method option '" << key << "'\n";
            return 4;
          }
        }
      }
      methods.push_back(cfg);
    }
  }
  if (methods.empty()) {
    std::cerr << "No methods parsed from '" << A.methods_csv << "'\n";
    return 4;
  }

  // Build human-readable labels (especially for SS) so the output reflects differentiating args
  std::vector<std::string> labels(methods.size());
  std::size_t max_label_len = 0;
  {
    // First handle SS with minimal distinguishing prefixes
    std::vector<std::size_t> ss_idxs;
    std::unordered_map<std::size_t, std::vector<std::string>> ss_components;
    auto policy_str = [](Policy p) {
      switch (p) {
        case Policy::Difficulty: return std::string("policy=difficulty");
        case Policy::Static: return std::string("policy=static");
      }
      return std::string("policy=unknown");
    };
    auto q_str = [](std::size_t qf) {
      if (qf <= 1) return std::string("q=n");
      return std::string("q=") + std::to_string(qf) + "n";
    };
    auto dbl = [](double v) {
      std::ostringstream oss;
      oss << std::setprecision(3) << std::fixed << v;
      return oss.str();
    };
    for (std::size_t i = 0; i < methods.size(); ++i) {
      if (methods[i].name != "ss") continue;
      ss_idxs.push_back(i);
      std::vector<std::string> comps;
      comps.push_back(policy_str(methods[i].policy));
      comps.push_back(q_str(methods[i].q_factor));
      if (methods[i].ss_per_item_eps) comps.push_back("eps=per-item");
      switch (methods[i].policy) {
        case Policy::Difficulty:
          comps.push_back("alphaReq=" + dbl(methods[i].alpha_req));
          comps.push_back("rM=" + dbl(methods[i].r_m));
          comps.push_back(std::string("mode=") + (methods[i].diff_mode == DifficultyMode::Predictive ? "predictive"
                                                   : (methods[i].diff_mode == DifficultyMode::ReactiveReq ? "reactive-req" : "reactive-eff")));
          comps.push_back(std::string("amb=") + (methods[i].ambiguity_adjust ? "on" : "off"));
          break;
        case Policy::Static:
          break;
      }
      ss_components[i] = std::move(comps);
    }

    // Choose minimal prefix that distinguishes each SS config
    std::unordered_map<std::string, int> base_counts;
    for (auto idx : ss_idxs) {
      const auto& comps = ss_components[idx];
      std::string chosen = comps.empty() ? std::string{} : comps.front();
      for (std::size_t L = 1; L <= comps.size(); ++L) {
        std::ostringstream oss;
        for (std::size_t k = 0; k < L; ++k) {
          if (k) oss << ' ';
          oss << comps[k];
        }
        auto candidate = oss.str();
        bool unique = true;
        for (auto other : ss_idxs) {
          if (other == idx) continue;
          const auto& ocomps = ss_components[other];
          if (ocomps.size() < L) continue;
          std::ostringstream oss2;
          for (std::size_t k = 0; k < L; ++k) {
            if (k) oss2 << ' ';
            oss2 << ocomps[k];
          }
          if (oss2.str() == candidate) { unique = false; break; }
        }
        if (unique) { chosen = candidate; break; }
        chosen = candidate; // fallback to longest if none unique
      }
      auto base = std::string("ss[") + chosen + "]";
      int& cnt = base_counts[base];
      std::string final = base;
      if (cnt > 0) final += "#" + std::to_string(cnt);
      ++cnt;
      labels[idx] = std::move(final);
    }

    // Other methods keep original unique numbering
    std::unordered_map<std::string, int> label_count;
    for (std::size_t i = 0; i < methods.size(); ++i) {
      if (!labels[i].empty()) continue; // already set (ss)
      int cnt = label_count[methods[i].name]++;
      labels[i] = (cnt == 0) ? methods[i].name : (methods[i].name + "#" + std::to_string(cnt));
    }

    for (const auto& lb : labels) {
      if (lb.size() > max_label_len) max_label_len = lb.size();
    }
  }



  // Build groups
  std::vector<MethodGroup> groups;
  std::vector<std::size_t> ss_q_caps(methods.size(), 0);
  std::vector<std::size_t> ss_q_curs(methods.size(), 0);
  std::vector<std::size_t> hybrid_qe(methods.size(), 0), hybrid_qa(methods.size(), 0);
  std::vector<sizing::PolicyState> policy_states(methods.size());
  std::vector<DifficultyVerification> pending_verify(methods.size());
  constexpr std::size_t HYB_HEAD_SLOT_BYTES = 24;
  constexpr std::size_t HYB_TAIL_SLOT_BYTES = 32;
  auto hybrid_tail_cap_for_head = [&](std::size_t q_e) -> std::size_t {
    const std::size_t head_bytes = q_e * HYB_HEAD_SLOT_BYTES;
    if (head_bytes >= bytes_per_part) return 1;
    return std::max<std::size_t>(std::size_t{1}, (bytes_per_part - head_bytes) / HYB_TAIL_SLOT_BYTES);
  };
  groups.reserve(methods.size());
  for (std::size_t idx=0; idx<methods.size(); ++idx) {
    const auto& spec = methods[idx];
    auto g = make_method(spec, A.m, bytes_per_part, A.n_param);
    g.label = labels[idx];
    groups.push_back(std::move(g));
    if (spec.name == "ss") {
      const std::size_t cap = std::max<std::size_t>(std::size_t(1), bytes_per_part / 32);
      ss_q_caps[idx] = cap;
      ss_q_curs[idx] = std::min<std::size_t>(A.n_param * spec.q_factor, cap);
    } else if (spec.name == "hybrid") {
      hybrid_qe[idx] = 0;          // no prior head knowledge in the first window
      const std::size_t cap = hybrid_tail_cap_for_head(hybrid_qe[idx]);
      ss_q_caps[idx] = cap; // reuse cap helper
      if (spec.tail_policy == TailPolicy::Static) {
        hybrid_qa[idx] = A.n_param * std::max<std::size_t>(1, spec.tail_q_factor);
      } else {
        hybrid_qa[idx] = A.n_param;  // start tail at n
      }
      hybrid_qa[idx] = std::max<std::size_t>(A.n_param, std::min<std::size_t>(hybrid_qa[idx], cap));
      // Reconfigure the freshly-built hybrid sketches to enforce the intended q_e/q_a
      for (auto& p : groups.back().parts) {
        auto hy = std::make_unique<HybridSS>(hybrid_qe[idx], hybrid_qa[idx]);
        Part* part = &p;
        hy->set_admission_callback([part](const Id128& id){
          auto it = part->recent.find(id);
          if (it == part->recent.end()) return;
          part->amap.bind_bytes(id, it->second);
        });
        hy->reconfigure(hybrid_qe[idx], hybrid_qa[idx], A.n_param, /*head_mass_frac=*/0.0);
        p.sketch = std::move(hy);
      }
    }
  }

  // Must include oracle reference
  bool has_oracle = false;
  for (auto& g : groups) if (g.name == "oracle") { has_oracle = true; break; }
  if (!has_oracle) {
    std::cerr << "ERROR: methods must include 'oracle' for reference.\n";
    return 3;
  }


  // Per-method per-window metrics
  std::unordered_map<std::string, std::vector<WindowMetrics>> perwin;
  std::unordered_map<std::string, double> mem_algo_equiv_per_part_sum;
  std::unordered_map<std::string, double> mem_key_equiv_per_part_sum;
  std::unordered_map<std::string, double> mem_worker_equiv_per_part_sum;
  std::unordered_map<std::string, double> mem_coord_input_flat_sum;
  std::unordered_map<std::string, double> mem_coord_work_flat_sum;
  std::unordered_map<std::string, double> mem_coord_peak_flat_sum;
  std::unordered_map<std::string, std::size_t> miss_req_cnt;
  std::unordered_map<std::string, std::size_t> miss_eff_cnt;
  std::unordered_map<std::string, std::size_t> miss_den_cnt;
  std::vector<std::uint64_t> update_events_by_group(groups.size(), 0);
  std::vector<double> update_ns_by_group(groups.size(), 0.0);
  const int labelw = static_cast<int>(max_label_len);

  // Ingestion state
  std::size_t curr_win = 0; bool first = false;
  std::size_t windows_flushed = 0;

  auto flush = [&](std::size_t win_idx){
    // End-of-window routine: evaluate all methods, print metrics, decide next window sizes, then reset/reseed state.
    // reduce each method
    GlobalResultLB oracleR{};
    std::vector<GlobalResultLB> Rs(groups.size());
    std::vector<ReduceTelemetry> coord_mem_by_group(groups.size());
    std::vector<std::size_t> coord_input_bytes_by_group(groups.size(), 0);
    std::vector<std::size_t> coord_peak_bytes_by_group(groups.size(), 0);
    std::vector<double> reduce_ms_by_group(groups.size(), 0.0);
    std::vector<double> control_ms_by_group(groups.size(), 0.0);
    std::vector<CsvRow> csv_rows(groups.size());

    // also keep snapshots for SS/hybrid policy (avoids recompute)
    std::vector<std::vector<SnapshotEx>> snaps_by_group(groups.size());

    for (std::size_t gi=0; gi<groups.size(); ++gi) {
      const auto& g = groups[gi];
      std::vector<SnapshotEx> snaps; snaps.reserve(g.parts.size());
      std::vector<const ArenaMap*> maps; maps.reserve(g.parts.size());
      std::vector<HLBucketSnapshot> snaps_hl; snaps_hl.reserve(g.parts.size());
      bool all_hl_bucketed = (g.name == "hl");
      for (const auto& p : g.parts) {
        snaps.push_back(p.sketch->snapshot_ex());
        maps.push_back(&p.amap);
        if (all_hl_bucketed) {
          auto* hl = dynamic_cast<const HeavyLocker*>(p.sketch.get());
          if (!hl) {
            all_hl_bucketed = false;
          } else {
            snaps_hl.push_back(hl->snapshot_bucketed());
          }
        }
      }
      snaps_by_group[gi] = std::move(snaps);

      GlobalResultLB R;
      const auto reduce_t0 = std::chrono::steady_clock::now();
      if (all_hl_bucketed && snaps_hl.size() == g.parts.size()) {
        std::size_t inb = sizeof(snaps_hl) + snaps_hl.capacity() * sizeof(decltype(snaps_hl)::value_type);
        for (const auto& s : snaps_hl) {
          inb += sizeof(s);
          inb += sizeof(s.candidates) + s.candidates.capacity() * sizeof(decltype(s.candidates)::value_type);
        }
        coord_input_bytes_by_group[gi] = inb;
        R = Coordinator::reduce_hl_bucketwise(snaps_hl, maps, A.n_param, &coord_mem_by_group[gi]);
      } else {
        const auto& sx = snaps_by_group[gi];
        std::size_t inb = sizeof(sx) + sx.capacity() * sizeof(SnapshotEx);
        for (const auto& s : sx) {
          inb += sizeof(s);
          inb += sizeof(s.candidates) + s.candidates.capacity() * sizeof(decltype(s.candidates)::value_type);
          inb += sizeof(s.errors) + s.errors.capacity() * sizeof(decltype(s.errors)::value_type);
        }
        coord_input_bytes_by_group[gi] = inb;
        R = Coordinator::reduce_global_with_lb(snaps_by_group[gi], maps, A.n_param, &coord_mem_by_group[gi]);
      }
      const auto reduce_t1 = std::chrono::steady_clock::now();
      reduce_ms_by_group[gi] =
          static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(reduce_t1 - reduce_t0).count()) / 1.0e6;
      coord_peak_bytes_by_group[gi] =
          coord_input_bytes_by_group[gi] + coord_mem_by_group[gi].total_peak_bytes;
      if (g.name == "oracle") oracleR = R;
      Rs[gi] = std::move(R);
    }

    // Evaluate vs oracle and print window header (print q for SS)
    std::cout << "Window " << win_idx << ":\n";
    auto sketch_mem_kib_for_group = [&](const MethodGroup& g) -> double {
      if (g.parts.empty()) return 0.0;
      double sum_bytes = 0.0;
      std::size_t cnt = 0;
      for (const auto& p : g.parts) {
        if (!p.sketch) continue;
        sum_bytes += static_cast<double>(p.sketch->memory_bytes());
        ++cnt;
      }
      if (cnt == 0) return 0.0;
      return (sum_bytes / static_cast<double>(cnt)) / 1024.0;
    };
    auto binding_mem_kib_for_group = [&](const MethodGroup& g) -> double {
      if (g.parts.empty()) return 0.0;
      double sum_bytes = 0.0;
      std::size_t cnt = 0;
      for (const auto& p : g.parts) {
        std::size_t part_bytes = p.amap.memory_bytes();
        part_bytes += p.recent.bucket_count() * sizeof(void*);
        part_bytes += p.recent.size() * sizeof(std::pair<const Id128, std::string>);
        for (const auto& kv : p.recent) part_bytes += kv.second.capacity();
        sum_bytes += static_cast<double>(part_bytes);
        ++cnt;
      }
      if (cnt == 0) return 0.0;
      return (sum_bytes / static_cast<double>(cnt)) / 1024.0;
    };

    // The first processed window is a warmup/adaptation seed. Do not infer this
    // from the numeric window ID, because input IDs are labels and may not start
    // at zero for externally supplied traces.
    const bool include_in_summary = (windows_flushed > 0);
    for (std::size_t gi=0; gi<Rs.size(); ++gi) {
      if (groups[gi].name == "oracle") continue;
      // if (groups[gi].name == "hybrid") {
      //   // One-time debug: compare hybrid vs oracle HH counts for the first window
      //   static bool dbg_hybrid_checked = false;
      //   if (!dbg_hybrid_checked) {
      //     dbg_hybrid_checked = true;
      //     std::unordered_map<Id128, std::uint64_t, Id128Hash> hyb_map;
      //     for (const auto& it : Rs[gi].items) hyb_map[it.id] = it.est;

      //     bool any = false;
      //     for (const auto& it : oracleR.items) {
      //       auto hm = hyb_map.find(it.id);
      //       if (hm == hyb_map.end() || hm->second != it.est) {
      //         if (!any) std::cerr << "[hybrid dbg] window " << win_idx << " HH diffs vs oracle:\n";
      //         any = true;
      //         std::cerr << "  id=" << Coordinator::id128_hex(it.id)
      //                   << " oracle=" << it.est
      //                   << " hybrid=" << (hm == hyb_map.end() ? 0 : hm->second) << "\n";
      //       }
      //     }
      //     if (!any) std::cerr << "[hybrid dbg] window " << win_idx << " hybrid matches oracle for all HH.\n";
      //   }
      // }
      auto wm = eval_vs_oracle(oracleR, Rs[gi], A.topk);
      if (include_in_summary) {
        perwin[groups[gi].label].push_back(wm);
      }
      std::cout << "  [" << std::left << std::setw(labelw) << groups[gi].label << "] " << std::right
                << "HH precision=" << std::fixed << std::setprecision(3)
                << wm.hh_precision << " recall=" << wm.hh_recall
                << " F1=" << wm.hh_f1
                << " AAE=" << std::setprecision(3) << wm.aae
                << " ARE=" << std::setprecision(3) << (wm.are * 100.0) << "%";
      if (groups[gi].name == "ss") std::cout << " q=" << ss_q_curs[gi];
      if (groups[gi].name == "hybrid") std::cout << " head=" << hybrid_qe[gi] << " tail=" << hybrid_qa[gi];
      const double sketch_mem_kib_win = sketch_mem_kib_for_group(groups[gi]);
      const double binding_mem_kib_win = binding_mem_kib_for_group(groups[gi]);
      const double algo_equiv_kib_win = sketch_mem_kib_win;
      const double key_equiv_kib_win = binding_mem_kib_win;
      const double worker_equiv_kib_win = algo_equiv_kib_win + key_equiv_kib_win;
      const double coord_input_flat_kib_win =
          static_cast<double>(coord_input_bytes_by_group[gi]) / 1024.0;
      const double coord_work_flat_kib_win =
          static_cast<double>(coord_mem_by_group[gi].total_peak_bytes) / 1024.0;
      const double coord_peak_flat_kib_win =
          static_cast<double>(coord_peak_bytes_by_group[gi]) / 1024.0;
      std::cout << "\tmem(wAlgo/wKey/wTotal)≈"
                << std::fixed << std::setprecision(2)
                << algo_equiv_kib_win << "/" << key_equiv_kib_win << "/" << worker_equiv_kib_win
                << " KiB"
                << "\tmem(cInput/cWork/cPeak)≈"
                << coord_input_flat_kib_win << "/" << coord_work_flat_kib_win << "/" << coord_peak_flat_kib_win
                << " KiB";
      if (wm.topk_overlap) {
        std::cout << " topk_overlap=" << std::setprecision(3) << *wm.topk_overlap;
      }
      std::cout << "\n";
      if (!A.csv_out.empty()) {
        auto& row = csv_rows[gi];
        row.window = win_idx;
        row.method = groups[gi].label;
        row.method_type = groups[gi].name;
        row.N_global = Rs[gi].N_global;
        row.threshold = Rs[gi].threshold;
        row.hh_precision = wm.hh_precision;
        row.hh_recall = wm.hh_recall;
        row.hh_f1 = wm.hh_f1;
        row.aae = wm.aae;
        row.are = wm.are;
        if (wm.topk_overlap) {
          row.has_topk = true;
          row.topk_overlap = *wm.topk_overlap;
        }
        if (groups[gi].name == "ss") row.q_current = ss_q_curs[gi];
        else if (groups[gi].name == "hybrid") {
          row.q_current = hybrid_qe[gi] + hybrid_qa[gi];
          row.q_head_current = hybrid_qe[gi];
          row.q_tail_current = hybrid_qa[gi];
        }
        row.cert = certification_metrics(oracleR, Rs[gi]);
        row.mem_worker_algo_kib = algo_equiv_kib_win;
        row.mem_worker_key_kib = key_equiv_kib_win;
        row.mem_worker_total_kib = worker_equiv_kib_win;
        row.mem_coord_input_kib = coord_input_flat_kib_win;
        row.mem_coord_work_kib = coord_work_flat_kib_win;
        row.mem_coord_peak_kib = coord_peak_flat_kib_win;
        row.update_events = update_events_by_group[gi];
        row.update_ms = update_ns_by_group[gi] / 1.0e6;
        row.update_mops = row.update_ms > 0.0
            ? (static_cast<double>(row.update_events) / (row.update_ms / 1000.0)) / 1.0e6
            : 0.0;
        row.reduce_ms = reduce_ms_by_group[gi];
      }
      // accumulate memory metrics per partition for averaging
      if (include_in_summary) {
        mem_algo_equiv_per_part_sum[groups[gi].label] += algo_equiv_kib_win;
        mem_key_equiv_per_part_sum[groups[gi].label] += key_equiv_kib_win;
        mem_worker_equiv_per_part_sum[groups[gi].label] += worker_equiv_kib_win;
        mem_coord_input_flat_sum[groups[gi].label] += coord_input_flat_kib_win;
        mem_coord_work_flat_sum[groups[gi].label] += coord_work_flat_kib_win;
        mem_coord_peak_flat_sum[groups[gi].label] += coord_peak_flat_kib_win;
      }
    }

    // If SS present, apply policy to compute q_next
    // Precompute hybrid seed + head mass fraction (min across partitions) and sizing rule for each hybrid
    std::vector<Id128> hybrid_seed_ids;
    for (std::size_t gi=0; gi<groups.size(); ++gi) {
      if (groups[gi].name != "hybrid") continue;
      const auto control_t0 = std::chrono::steady_clock::now();
      double hybrid_head_frac_min = 0.0;
      const auto& snaps_for_hybrid = snaps_by_group[gi];
      if (!snaps_for_hybrid.empty()) {
        double min_frac = 1.0;
        for (const auto& s : snaps_for_hybrid) {
          if (s.N_local == 0) { min_frac = 0.0; break; }
          const double frac = static_cast<double>(s.head_mass) / static_cast<double>(s.N_local);
          min_frac = std::min(min_frac, frac);
        }
        hybrid_head_frac_min = min_frac;
      }

      const auto& R_for_hybrid = Rs[gi];
      const auto& hyb_cfg = methods[gi];
      // Head seed from previous-window telemetry.
      hybrid_seed_ids.clear();
      std::size_t topk = std::min<std::size_t>(A.n_param, R_for_hybrid.items.size());
      std::size_t top2k = std::min<std::size_t>(2 * A.n_param, R_for_hybrid.items.size());
      for (std::size_t i = 0; i < R_for_hybrid.items.size(); ++i) {
        const auto& it = R_for_hybrid.items[i];
        const bool in_top = (i < topk);
        const bool in_top2 = (i < top2k);
        const bool frontier = (it.cert_ub >= R_for_hybrid.threshold);
        const bool challenger = (!in_top && frontier);
        if (hyb_cfg.head_policy == HeadPolicy::TopN) {
          if (in_top) hybrid_seed_ids.push_back(it.id);
        } else if (hyb_cfg.head_policy == HeadPolicy::Top2N) {
          if (in_top2) hybrid_seed_ids.push_back(it.id);
        } else if (hyb_cfg.head_policy == HeadPolicy::Threshold) {
          if (challenger) hybrid_seed_ids.push_back(it.id); // ub >= threshold
        } else { // Candidate
          if (in_top || challenger) hybrid_seed_ids.push_back(it.id);
        }
      }

      // Build residual-only telemetry for hybrid tail sizing (exclude promoted head keys).
      std::unordered_set<Id128, Id128Hash> hybrid_head_ids(hybrid_seed_ids.begin(), hybrid_seed_ids.end());
      std::vector<SnapshotEx> snaps_residual;
      snaps_residual.reserve(snaps_for_hybrid.size());
      for (const auto& s : snaps_for_hybrid) {
        SnapshotEx sr{};
        // Keep global-mass normalization for residual certificates/sizing.
        // Residual counts stay in candidates, but denominators remain N^t per paper.
        sr.N_local = s.N_local;
        sr.q_local = s.q_local;
        sr.head_mass = 0;
        sr.head_size = 0;
        sr.candidates.reserve(s.candidates.size());
        for (const auto& c : s.candidates) {
          if (hybrid_head_ids.find(c.id) != hybrid_head_ids.end()) continue;
          sr.candidates.push_back(c);
        }
        snaps_residual.push_back(std::move(sr));
      }
      std::vector<const ArenaMap*> maps_residual;
      maps_residual.reserve(groups[gi].parts.size());
      for (const auto& p : groups[gi].parts) maps_residual.push_back(&p.amap);
      const auto R_for_tail = Coordinator::reduce_global_with_lb(snaps_residual, maps_residual, A.n_param);
      std::unordered_set<Id128, Id128Hash> hybrid_tail_candidate_ids;
      const std::size_t topk_tail = std::min<std::size_t>(A.n_param, R_for_tail.items.size());
      for (std::size_t i = 0; i < R_for_tail.items.size(); ++i) {
        const auto& it = R_for_tail.items[i];
        const bool in_top = (i < topk_tail);
        const bool challenger = (!in_top && it.cert_ub >= R_for_tail.threshold);
        if (in_top || challenger) hybrid_tail_candidate_ids.insert(it.id);
      }

      hybrid_qe[gi] = std::max<std::size_t>(std::size_t{1}, hybrid_seed_ids.size()); // q_e = |C|
      const std::size_t tail_cap = hybrid_tail_cap_for_head(hybrid_qe[gi]);
      // Tail sizing: policy with head coverage deduction
      const double P_E_min = std::clamp(hybrid_head_frac_min, 0.0, 1.0); // min head mass fraction across partitions
      const std::size_t tail_floor = std::max<std::size_t>(
          std::size_t{1},
          static_cast<std::size_t>(std::ceil(static_cast<double>(A.n_param) * (1.0 - P_E_min))));
      if (hyb_cfg.tail_policy == TailPolicy::Static) {
        // Static: keep tail fixed to n_param (clamped to memory cap) instead of adapting with head coverage
        const std::size_t q_static = A.n_param * std::max<std::size_t>(1, hyb_cfg.tail_q_factor);
        hybrid_qa[gi] = std::max<std::size_t>(tail_floor, std::min<std::size_t>(q_static, tail_cap));
      } else {
        sizing::PolicyConfig cfg_h;
        cfg_h.n_param = A.n_param;
        cfg_h.r = hyb_cfg.r;
        cfg_h.alpha_req = hyb_cfg.alpha_req;
        cfg_h.delta_m = hyb_cfg.delta_m;
        cfg_h.residual_guard_window = hyb_cfg.res_guard_window;
        cfg_h.residual_guard_decay = hyb_cfg.res_guard_decay;
        cfg_h.symmetric_relaxation = hyb_cfg.symmetric_relaxation;
        cfg_h.censored_control = hyb_cfg.censored_control;
        cfg_h.probe_residual_guard = hyb_cfg.probe_residual_guard;
        cfg_h.probe_strategy = hyb_cfg.probe_strategy;
        cfg_h.probe_pressure_gate = hyb_cfg.probe_pressure_gate;
        cfg_h.ambiguity_adjust = hyb_cfg.ambiguity_adjust;
        cfg_h.r_m = hyb_cfg.r_m;
        cfg_h.q_cur = hybrid_qa[gi];
        cfg_h.q_min = tail_floor;
        cfg_h.q_cap = tail_cap;
        cfg_h.q_max = tail_cap;
        cfg_h.candidate_ids = &hybrid_tail_candidate_ids;
        switch (hyb_cfg.tail_policy) {
          case TailPolicy::Difficulty: cfg_h.kind = sizing::PolicyKind::difficulty; break;
          case TailPolicy::Static: cfg_h.kind = sizing::PolicyKind::fixed; break;
        }
        auto res_h = sizing::next_q_ss(R_for_tail, snaps_residual, cfg_h, &policy_states[gi]);
        const std::size_t q_req_resid = std::max<std::size_t>(
            tail_floor,
            res_h.q_req);
        const std::size_t q_eff_resid = std::max<std::size_t>(
            tail_floor,
            res_h.q_base);
        const std::size_t q_req_fair = std::min(q_req_resid, tail_cap);
        const std::size_t q_eff_fair = std::min(q_eff_resid, tail_cap);
        if (include_in_summary) {
          miss_den_cnt[groups[gi].label] += 1;
          if (hybrid_qa[gi] < q_req_fair) miss_req_cnt[groups[gi].label] += 1;
          if (hybrid_qa[gi] < q_eff_fair) miss_eff_cnt[groups[gi].label] += 1;
        }
        if (hyb_cfg.tail_policy == TailPolicy::Difficulty) {
          std::size_t q_next_mode = res_h.q_next;
          if (hyb_cfg.diff_mode == DifficultyMode::ReactiveReq) q_next_mode = std::max<std::size_t>(A.n_param, res_h.q_req);
          else if (hyb_cfg.diff_mode == DifficultyMode::ReactiveEff) q_next_mode = std::max<std::size_t>(A.n_param, res_h.q_base);
          if (cfg_h.q_cap != std::size_t(-1)) q_next_mode = std::min(q_next_mode, cfg_h.q_cap);
          q_next_mode = std::max(q_next_mode, cfg_h.q_min);
          if (cfg_h.q_max != std::size_t(-1)) q_next_mode = std::min(q_next_mode, cfg_h.q_max);
          res_h.q_next = q_next_mode;
        }
        if (hyb_cfg.tail_policy == TailPolicy::Difficulty && pending_verify[gi].valid) {
          const std::size_t q_req_resid = std::max<std::size_t>(
              tail_floor,
              res_h.q_req);
          const std::size_t q_eff_resid = std::max<std::size_t>(
              tail_floor,
              res_h.q_base);
          const std::size_t q_pred_eff_resid = std::max<std::size_t>(
              tail_floor,
              static_cast<std::size_t>(std::llround(pending_verify[gi].tilde_qpred)));
          const bool service_ok = (q_req_resid <= pending_verify[gi].q_planned);
          const bool control_ok = (q_eff_resid <= pending_verify[gi].q_planned);
          const bool calib_eff_ok = (q_eff_resid <= q_pred_eff_resid);
          std::cout << "  [" << std::left << std::setw(labelw) << groups[gi].label << "] " << std::right
                    << "verify: qBaseline_resid=" << q_req_resid
                    << " qActuation_resid=" << q_eff_resid
                    << " | pred(qActuation_resid<=" << q_pred_eff_resid
                    << ", q=" << pending_verify[gi].q_planned << ")"
                    << " | service=" << (service_ok ? "OK" : "FAIL")
                    << " control=" << (control_ok ? "OK" : "FAIL")
                    << " calibEff=" << (calib_eff_ok ? "OK" : "FAIL") << "\n";
        }
        if (hyb_cfg.tail_policy == TailPolicy::Difficulty) {
          pending_verify[gi].valid = true;
          pending_verify[gi].tilde_qpred = res_h.q_pred_tilde;
        }
        if (!A.csv_out.empty()) {
          auto& row = csv_rows[gi];
          row.q_req = q_req_resid;
          row.q_req_unclipped = res_h.q_req_unclipped;
          row.margin_alpha = res_h.margin_alpha;
          row.service_violation = res_h.service_violation ? 1 : 0;
          row.q_up = res_h.q_up;
          row.q_baseline = res_h.q_baseline;
          row.probe_issued = res_h.probe_issued ? 1 : 0;
          row.probe_failed = res_h.probe_failed ? 1 : 0;
          row.q_eff_replay = q_eff_resid;
          row.q_eff_pred = res_h.q_pred;
          row.q_eff_pred_tilde = res_h.q_pred_tilde;
          row.miss_req = hybrid_qa[gi] < q_req_fair ? 1 : 0;
          row.miss_eff = hybrid_qa[gi] < q_eff_fair ? 1 : 0;
          row.over_req = q_req_resid == 0 ? NAN : static_cast<double>(hybrid_qa[gi]) / static_cast<double>(q_req_resid);
          row.over_eff = q_eff_resid == 0 ? NAN : static_cast<double>(hybrid_qa[gi]) / static_cast<double>(q_eff_resid);
          row.calib_bias = res_h.b_q;
        }
        // Tail controller already operates on residual telemetry; deploy q_next directly.
        hybrid_qa[gi] = std::max<std::size_t>(tail_floor, std::min<std::size_t>(res_h.q_next, tail_cap));
        if (hyb_cfg.tail_policy == TailPolicy::Difficulty) {
          // q_planned must match the actually deployed residual tail size.
          pending_verify[gi].q_planned = hybrid_qa[gi];
        }
        if (hyb_cfg.tail_policy == TailPolicy::Difficulty) {
          std::cout << "  [" << std::left << std::setw(labelw) << groups[gi].label << "] "
                    << std::right << "hyb_tail difficulty:"
                    << " mode=" << (hyb_cfg.diff_mode == DifficultyMode::Predictive ? "predictive"
                                    : (hyb_cfg.diff_mode == DifficultyMode::ReactiveReq ? "reactive-req" : "reactive-eff"))
                    << " marginAlpha=" << std::setprecision(6) << res_h.margin_alpha
                    << " violation=" << (res_h.service_violation ? "yes" : "no")
                    << " qUp=" << res_h.q_up
                    << " qBaseline=" << res_h.q_baseline
                    << " qActuation=" << res_h.q_base
                    << " probe=" << (res_h.probe_issued ? "issued"
                                     : (res_h.probe_failed ? "failed" : "none"))
                    << "\n";
        }
      }
      if (!A.csv_out.empty()) {
        csv_rows[gi].q_next = hybrid_qe[gi] + hybrid_qa[gi];
        csv_rows[gi].q_head_next = hybrid_qe[gi];
        csv_rows[gi].q_tail_next = hybrid_qa[gi];
      }
      const auto control_t1 = std::chrono::steady_clock::now();
      control_ms_by_group[gi] =
          static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(control_t1 - control_t0).count()) / 1.0e6;
      if (!A.csv_out.empty()) csv_rows[gi].control_ms = control_ms_by_group[gi];
      // Reset/reseed only this hybrid group
      for (auto& p : groups[gi].parts) {
        p.sketch->reset_window();
        p.amap.clear();
        p.recent.clear();
        auto hy = std::make_unique<HybridSS>(hybrid_qe[gi], hybrid_qa[gi]);
        Part* part = &p;
        hy->set_admission_callback([part](const Id128& id){
          auto it = part->recent.find(id);
          if (it == part->recent.end()) return;
          part->amap.bind_bytes(id, it->second);
        });
        hy->reconfigure(hybrid_qe[gi], hybrid_qa[gi], A.n_param, hybrid_head_frac_min);
        hy->seed_head(hybrid_seed_ids);
        p.sketch = std::move(hy);
      }
    }

    for (std::size_t gi=0; gi<groups.size(); ++gi) {
      if (groups[gi].name != "ss") continue;
      const auto control_t0 = std::chrono::steady_clock::now();

      sizing::PolicyConfig cfg{};
      const auto& ss_cfg = methods[gi];
      switch (ss_cfg.policy) {
        case Policy::Difficulty: cfg.kind = sizing::PolicyKind::difficulty; break;
        case Policy::Static: cfg.kind = sizing::PolicyKind::fixed; break;
      }
      cfg.n_param = A.n_param;
      cfg.r = ss_cfg.r;
      cfg.alpha_req = ss_cfg.alpha_req;
      cfg.delta_m = ss_cfg.delta_m;
      cfg.residual_guard_window = ss_cfg.res_guard_window;
      cfg.residual_guard_decay = ss_cfg.res_guard_decay;
      cfg.symmetric_relaxation = ss_cfg.symmetric_relaxation;
      cfg.censored_control = ss_cfg.censored_control;
      cfg.probe_residual_guard = ss_cfg.probe_residual_guard;
      cfg.probe_strategy = ss_cfg.probe_strategy;
      cfg.probe_pressure_gate = ss_cfg.probe_pressure_gate;
      cfg.ambiguity_adjust = ss_cfg.ambiguity_adjust;
      cfg.r_m = ss_cfg.r_m;
      cfg.q_cur = (ss_cfg.policy == Policy::Static) ? (A.n_param * ss_cfg.q_factor) : ss_q_curs[gi];
      cfg.q_cap = ss_q_caps[gi];
      cfg.q_min = (ss_cfg.policy == Policy::Static) ? (A.n_param * ss_cfg.q_factor) : A.n_param;
      cfg.q_max = ss_q_caps[gi];
      std::unordered_set<Id128, Id128Hash> ss_candidate_ids;
      ss_candidate_ids.reserve(std::min<std::size_t>(Rs[gi].items.size(), A.n_param * 2));
      const std::size_t topk = std::min<std::size_t>(A.n_param, Rs[gi].items.size());
      for (std::size_t i = 0; i < Rs[gi].items.size(); ++i) {
        const auto& it = Rs[gi].items[i];
        const bool in_top = (i < topk);
        const bool challenger = (!in_top && it.cert_ub >= Rs[gi].threshold);
        if (in_top || challenger) ss_candidate_ids.insert(it.id);
      }
      cfg.candidate_ids = &ss_candidate_ids;
      auto res = sizing::next_q_ss(Rs[gi], snaps_by_group[gi], cfg, &policy_states[gi]);
      if (include_in_summary && ss_cfg.policy == Policy::Difficulty) {
        const std::size_t q_req_fair = std::min(res.q_req, ss_q_caps[gi]);
        const std::size_t q_eff_fair = std::min(res.q_base, ss_q_caps[gi]);
        miss_den_cnt[groups[gi].label] += 1;
        if (ss_q_curs[gi] < q_req_fair) miss_req_cnt[groups[gi].label] += 1;
        if (ss_q_curs[gi] < q_eff_fair) miss_eff_cnt[groups[gi].label] += 1;
      }
      if (cfg.kind == sizing::PolicyKind::difficulty) {
        std::size_t q_next_mode = res.q_next;
        if (ss_cfg.diff_mode == DifficultyMode::ReactiveReq) q_next_mode = std::max<std::size_t>(A.n_param, res.q_req);
        else if (ss_cfg.diff_mode == DifficultyMode::ReactiveEff) q_next_mode = std::max<std::size_t>(A.n_param, res.q_base);
        if (cfg.q_cap != std::size_t(-1)) q_next_mode = std::min(q_next_mode, cfg.q_cap);
        q_next_mode = std::max(q_next_mode, cfg.q_min);
        if (cfg.q_max != std::size_t(-1)) q_next_mode = std::min(q_next_mode, cfg.q_max);
        res.q_next = q_next_mode;
      }
      if (ss_cfg.policy == Policy::Difficulty && pending_verify[gi].valid) {
        const bool service_ok = (res.q_req <= pending_verify[gi].q_planned);
        const bool control_ok = (res.q_base <= pending_verify[gi].q_planned);
        const bool calib_eff_ok = (res.q_base <= static_cast<std::size_t>(std::llround(pending_verify[gi].tilde_qpred)));
        std::cout << "  [" << std::left << std::setw(labelw) << groups[gi].label << "] " << std::right
                  << "verify: qBaseline=" << res.q_req
                  << " qActuation=" << res.q_base
                  << " | pred(qActuation<=" << static_cast<std::size_t>(std::llround(pending_verify[gi].tilde_qpred))
                  << ", q=" << pending_verify[gi].q_planned << ")"
                  << " | service=" << (service_ok ? "OK" : "FAIL")
                  << " control=" << (control_ok ? "OK" : "FAIL")
                  << " calibEff=" << (calib_eff_ok ? "OK" : "FAIL") << "\n";
      }
      if (ss_cfg.policy == Policy::Difficulty) {
        pending_verify[gi].valid = true;
        pending_verify[gi].tilde_qpred = res.q_pred_tilde;
        pending_verify[gi].q_planned = res.q_next;
      }

      if (!A.csv_out.empty()) {
        auto& row = csv_rows[gi];
        row.q_next = res.q_next;
        if (ss_cfg.policy == Policy::Difficulty) {
          const std::size_t q_req_fair = std::min(res.q_req, ss_q_caps[gi]);
          const std::size_t q_eff_fair = std::min(res.q_base, ss_q_caps[gi]);
          row.q_req = res.q_req;
          row.q_req_unclipped = res.q_req_unclipped;
          row.margin_alpha = res.margin_alpha;
          row.service_violation = res.service_violation ? 1 : 0;
          row.q_up = res.q_up;
          row.q_baseline = res.q_baseline;
          row.probe_issued = res.probe_issued ? 1 : 0;
          row.probe_failed = res.probe_failed ? 1 : 0;
          row.q_eff_replay = res.q_base;
          row.q_eff_pred = res.q_pred;
          row.q_eff_pred_tilde = res.q_pred_tilde;
          row.miss_req = ss_q_curs[gi] < q_req_fair ? 1 : 0;
          row.miss_eff = ss_q_curs[gi] < q_eff_fair ? 1 : 0;
          row.over_req = res.q_req == 0 ? NAN : static_cast<double>(ss_q_curs[gi]) / static_cast<double>(res.q_req);
          row.over_eff = res.q_base == 0 ? NAN : static_cast<double>(ss_q_curs[gi]) / static_cast<double>(res.q_base);
          row.calib_bias = res.b_q;
        }
      }

      std::cout << std::fixed;
      std::cout << "  [" << std::left << std::setw(labelw) << groups[gi].label << "] " << std::right
                << "q_next=" << res.q_next << " (policy=";
      switch (cfg.kind) {
        case sizing::PolicyKind::difficulty:
          {
          std::cout << "difficulty, alphaReq=" << ss_cfg.alpha_req
                    << ", ambAdjust=" << (ss_cfg.ambiguity_adjust ? "on" : "off")
                    << ", mode=" << (ss_cfg.diff_mode == DifficultyMode::Predictive ? "predictive"
                                     : (ss_cfg.diff_mode == DifficultyMode::ReactiveReq ? "reactive-req" : "reactive-eff"))
                    << ", marginAlpha=" << std::setprecision(6) << res.margin_alpha
                    << ", violation=" << (res.service_violation ? "yes" : "no")
                    << ", qUp=" << res.q_up
                    << ", qBaseline=" << res.q_baseline
                    << ", qActuation=" << res.q_base
                    << ", probe=" << (res.probe_issued ? "issued"
                                       : (res.probe_failed ? "failed" : "none"));
          break;
          }
        default: std::cout << "fixed";
      }
      std::cout << ")\n";

      const auto control_t1 = std::chrono::steady_clock::now();
      control_ms_by_group[gi] =
          static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(control_t1 - control_t0).count()) / 1.0e6;
      if (!A.csv_out.empty()) csv_rows[gi].control_ms = control_ms_by_group[gi];

      // Reset and rebuild only this SS group
      const std::size_t next_q = (ss_cfg.policy == Policy::Static) ? (A.n_param * ss_cfg.q_factor) : res.q_next;
      ss_q_curs[gi] = next_q;
      for (auto& p : groups[gi].parts) {
        p.sketch->reset_window();
        p.amap.clear();
        p.recent.clear();
      }
      rebuild_ss_group(groups[gi], ss_q_curs[gi], methods[gi].ss_per_item_eps);
    }

    // Reset non-SS/non-hybrid state
    for (std::size_t gi=0; gi<groups.size(); ++gi) {
      if (groups[gi].name == "ss" || groups[gi].name == "hybrid") continue;
      for (auto& p : groups[gi].parts) {
        p.sketch->reset_window();
        p.amap.clear();
        p.recent.clear();
      }
    }

    if (!A.csv_out.empty()) {
      for (std::size_t gi = 0; gi < groups.size(); ++gi) {
        if (groups[gi].name == "oracle") continue;
        write_csv_row(csv_file, csv_rows[gi]);
      }
    }

    std::fill(update_events_by_group.begin(), update_events_by_group.end(), 0);
    std::fill(update_ns_by_group.begin(), update_ns_by_group.end(), 0.0);
    ++windows_flushed;
  };

  // Ingest once, feed all methods identically
  JsonGzNestedReader::read(A.path, [&](std::size_t win, std::size_t part, const std::string& raw, int w){
    const std::size_t pidx = part % A.m;
    if (!first) { curr_win = win; first = true; }
    if (win != curr_win) { flush(curr_win); curr_win = win; }

    const Id128 id = id128_for(raw);

    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
      // admission-only binding for all sketches (fairness policy)
      auto& g = groups[gi];
      g.parts[pidx].recent[id] = raw;
      const auto t0 = std::chrono::steady_clock::now();
      g.parts[pidx].sketch->update(id, w);
      const auto t1 = std::chrono::steady_clock::now();
      update_ns_by_group[gi] +=
          static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
      update_events_by_group[gi] += 1;
    }
  });

  if (first) flush(curr_win);

  // Aggregate & print summary
  std::cout << "\n=== Summary vs ORACLE (HH by default"
            << (A.topk ? ", plus top-k overlap" : "") << ") ===\n";
  std::cout << "(memory fairness default: workerEq is per-partition mean; coordinator is flat per-window average; coordPeakEq=coordInputEq+coordWorkEq)\n";

  auto fmt = [](double v, int prec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
  };

  // stable order based on declared groups (excluding oracle), to ensure all methods are reported
  std::unordered_set<std::string> seen;
  struct SummaryRow {
    std::string label, win, prec, rec, f1, aae, are, miss_req, miss_eff, topk;
    std::string mem_algo_eq, mem_key_eq, mem_worker_eq;
    std::string mem_coord_input_eq, mem_coord_work_eq, mem_coord_peak_eq;
  };
  std::vector<SummaryRow> rows;
  std::size_t w_label = max_label_len, w_win = 0, w_prec = 0, w_rec = 0, w_f1 = 0, w_aae = 0, w_are = 0, w_mreq = 0, w_meff = 0, w_topk = 0;
  std::size_t w_mem_aeq = 0, w_mem_keq = 0, w_mem_weq = 0, w_mem_cieq = 0, w_mem_cweq = 0, w_mem_cpeq = 0;

  for (const auto& g : groups) {
    if (g.name == "oracle") continue;
    if (!seen.insert(g.label).second) continue;
    auto it = perwin.find(g.label);
    if (it == perwin.end()) continue;
    auto ag = mean_of(it->second);

    SummaryRow r;
    r.label = g.label;
    r.win   = std::to_string(ag.windows);
    r.prec  = fmt(ag.hh_precision_avg, 3);
    r.rec   = fmt(ag.hh_recall_avg, 3);
    r.f1    = fmt(ag.hh_f1_avg, 3);
    r.aae   = fmt(ag.aae_avg, 3);
    r.are   = fmt(ag.are_avg * 100.0, 3);
    auto mem_ae_it = mem_algo_equiv_per_part_sum.find(g.label);
    auto mem_ke_it = mem_key_equiv_per_part_sum.find(g.label);
    auto mem_we_it = mem_worker_equiv_per_part_sum.find(g.label);
    auto mem_cie_it = mem_coord_input_flat_sum.find(g.label);
    auto mem_cwe_it = mem_coord_work_flat_sum.find(g.label);
    auto mem_cpe_it = mem_coord_peak_flat_sum.find(g.label);
    if (ag.windows > 0) {
      const double avg_ae = (mem_ae_it != mem_algo_equiv_per_part_sum.end()) ? (mem_ae_it->second / static_cast<double>(ag.windows)) : 0.0;
      const double avg_ke = (mem_ke_it != mem_key_equiv_per_part_sum.end()) ? (mem_ke_it->second / static_cast<double>(ag.windows)) : 0.0;
      r.mem_algo_eq = fmt(avg_ae, 2);
      r.mem_key_eq = fmt(avg_ke, 2);
      if (mem_we_it != mem_worker_equiv_per_part_sum.end()) r.mem_worker_eq = fmt(mem_we_it->second / static_cast<double>(ag.windows), 2);
      if (mem_cie_it != mem_coord_input_flat_sum.end()) r.mem_coord_input_eq = fmt(mem_cie_it->second / static_cast<double>(ag.windows), 2);
      if (mem_cwe_it != mem_coord_work_flat_sum.end()) r.mem_coord_work_eq = fmt(mem_cwe_it->second / static_cast<double>(ag.windows), 2);
      if (mem_cpe_it != mem_coord_peak_flat_sum.end()) r.mem_coord_peak_eq = fmt(mem_cpe_it->second / static_cast<double>(ag.windows), 2);
    }
    if (ag.topk_overlap_avg) {
      r.topk = fmt(*ag.topk_overlap_avg, 3);
    }
    auto den_it = miss_den_cnt.find(g.label);
    if (den_it != miss_den_cnt.end() && den_it->second > 0) {
      const double miss_req = 100.0 * static_cast<double>(miss_req_cnt[g.label]) / static_cast<double>(den_it->second);
      const double miss_eff = 100.0 * static_cast<double>(miss_eff_cnt[g.label]) / static_cast<double>(den_it->second);
      r.miss_req = fmt(miss_req, 2) + "%";
      r.miss_eff = fmt(miss_eff, 2) + "%";
    }

    w_win   = std::max(w_win,   r.win.size());
    w_prec  = std::max(w_prec,  r.prec.size());
    w_rec   = std::max(w_rec,   r.rec.size());
    w_f1    = std::max(w_f1,    r.f1.size());
    w_aae   = std::max(w_aae,   r.aae.size());
    w_are   = std::max(w_are,   r.are.size());
    w_mreq  = std::max(w_mreq,  r.miss_req.size());
    w_meff  = std::max(w_meff,  r.miss_eff.size());
    w_mem_aeq = std::max(w_mem_aeq, r.mem_algo_eq.size());
    w_mem_keq = std::max(w_mem_keq, r.mem_key_eq.size());
    w_mem_weq = std::max(w_mem_weq, r.mem_worker_eq.size());
    w_mem_cieq = std::max(w_mem_cieq, r.mem_coord_input_eq.size());
    w_mem_cweq = std::max(w_mem_cweq, r.mem_coord_work_eq.size());
    w_mem_cpeq = std::max(w_mem_cpeq, r.mem_coord_peak_eq.size());
    w_topk  = std::max(w_topk,  r.topk.size());

    rows.push_back(std::move(r));
  }

  for (const auto& r : rows) {
    std::cout << std::left << std::setw(w_label) << r.label << std::right
              << "\twindows=" << std::setw(w_win) << r.win
              << "\tHH: precision=" << std::setw(w_prec) << r.prec
              << " recall=" << std::setw(w_rec) << r.rec
              << " F1=" << std::setw(w_f1) << r.f1
              << "\tAAE=" << std::setw(w_aae) << r.aae
              << "\tARE=" << std::setw(w_are) << r.are << "%";
    if (!r.topk.empty()) {
      std::cout << "\ttopK_overlap=" << std::setw(w_topk) << r.topk;
    }
    if (!r.mem_worker_eq.empty()) {
      std::cout << "\tmem(wAlgo/wKey/wTotal)≈"
                << std::setw(w_mem_aeq) << r.mem_algo_eq << "/"
                << std::setw(w_mem_keq) << r.mem_key_eq << "/"
                << std::setw(w_mem_weq) << r.mem_worker_eq << " KiB"
                << "\tmem(cInput/cWork/cPeak)≈"
                << std::setw(w_mem_cieq) << r.mem_coord_input_eq << "/"
                << std::setw(w_mem_cweq) << r.mem_coord_work_eq << "/"
                << std::setw(w_mem_cpeq) << r.mem_coord_peak_eq << " KiB";
    }
    if (!r.miss_req.empty()) {
      std::cout << "\tmissReq=" << std::setw(w_mreq) << r.miss_req
                << " missEff=" << std::setw(w_meff) << r.miss_eff;
    }
    std::cout << "\n";
  }

  return 0;
}
