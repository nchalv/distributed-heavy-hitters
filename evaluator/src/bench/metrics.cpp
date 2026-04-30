#include "hh/bench/metrics.hpp"
#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace hh {

static std::unordered_set<Id128, Id128Hash>
hh_set(const GlobalResultLB& R) {
  std::unordered_set<Id128, Id128Hash> s;
  for (auto& it : R.items) if (it.est >= R.threshold) s.insert(it.id);
  return s;
}

static std::vector<Id128>
topk_ids(const GlobalResultLB& R, std::size_t k) {
  std::vector<Id128> out;
  out.reserve(std::min(k, R.items.size()));
  for (std::size_t i = 0; i < R.items.size() && out.size() < k; ++i) {
    out.push_back(R.items[i].id);
  }
  return out;
}

WindowMetrics eval_vs_oracle(const GlobalResultLB& oracle,
                             const GlobalResultLB& method,
                             std::optional<std::size_t> k_opt)
{
  WindowMetrics wm{}; wm.N_global = oracle.N_global;

  // Build quick lookup for method estimates
  std::unordered_map<Id128, std::uint64_t, Id128Hash> meth_est;
  meth_est.reserve(method.items.size() * 2 + 8);
  for (const auto& it : method.items) meth_est[it.id] = it.est;

  // HH precision/recall by default
  auto H_oracle  = hh_set(oracle);
  auto H_method = hh_set(method);

  std::size_t tp = 0;
  for (auto& id : H_method) if (H_oracle.count(id)) ++tp;

  const std::size_t fp = (H_method.size() >= tp) ? (H_method.size() - tp) : 0;
  const std::size_t fn = (H_oracle.size() >= tp) ? (H_oracle.size() - tp) : 0;

  wm.hh_precision = H_method.empty() ? 1.0 : (double)tp / (double)H_method.size();
  wm.hh_recall    = H_oracle.empty()  ? 1.0 : (double)tp / (double)H_oracle.size();
  const double denom = wm.hh_precision + wm.hh_recall;
  wm.hh_f1 = (denom > 0.0) ? (2.0 * wm.hh_precision * wm.hh_recall) / denom : 0.0;
  wm.hh_tp = tp;
  wm.hh_fp = fp;
  wm.hh_fn = fn;

  // AAE/ARE over exact items
  double abs_err_sum = 0.0;
  double rel_err_sum = 0.0;
  std::size_t err_cnt = 0;
  for (const auto& ex : oracle.items) {
    if (ex.est < oracle.threshold) continue; // only heavy hitters
    const double ex_val = static_cast<double>(ex.est);
    const double meth_val = static_cast<double>(meth_est.count(ex.id) ? meth_est[ex.id] : 0);
    const double diff = std::fabs(meth_val - ex_val);
    abs_err_sum += diff;
    if (ex_val > 0.0) rel_err_sum += diff / ex_val;
    ++err_cnt;
  }
  wm.abs_err_sum = abs_err_sum;
  wm.rel_err_sum = rel_err_sum;
  wm.hh_items = err_cnt;
  wm.aae = (err_cnt == 0) ? 0.0 : abs_err_sum / static_cast<double>(err_cnt);
  wm.are = (err_cnt == 0) ? 0.0 : rel_err_sum / static_cast<double>(err_cnt);

  // Optional top-k overlap
  if (k_opt && *k_opt > 0) {
    auto K_exact  = topk_ids(oracle, *k_opt);
    auto K_method = topk_ids(method, *k_opt);
    std::unordered_set<Id128, Id128Hash> Kset(K_exact.begin(), K_exact.end());
    std::size_t inter = 0;
    for (auto& id : K_method) if (Kset.count(id)) ++inter;

    wm.k_used = std::min({*k_opt, K_exact.size(), K_method.size()});
    wm.topk_overlap = (wm.k_used == 0) ? std::optional<double>(1.0)
                                       : std::optional<double>((double)inter / (double)wm.k_used);
  }
  return wm;
}

AggregateMetrics mean_of(const std::vector<WindowMetrics>& ws) {
  AggregateMetrics ag{}; ag.windows = ws.size();
  if (ws.empty()) return ag;
  double tp_sum = 0.0, fp_sum = 0.0, fn_sum = 0.0;
  double abs_err_sum = 0.0, rel_err_sum = 0.0;
  std::size_t hh_items = 0;
  double topk_weighted = 0.0; std::size_t topk_total = 0;

  for (auto& w : ws) {
    tp_sum += static_cast<double>(w.hh_tp);
    fp_sum += static_cast<double>(w.hh_fp);
    fn_sum += static_cast<double>(w.hh_fn);
    abs_err_sum += w.abs_err_sum;
    rel_err_sum += w.rel_err_sum;
    hh_items += w.hh_items;
    if (w.topk_overlap && w.k_used > 0) {
      topk_weighted += (*w.topk_overlap) * static_cast<double>(w.k_used);
      topk_total += w.k_used;
    }
  }

  const double prec_denom = tp_sum + fp_sum;
  const double rec_denom  = tp_sum + fn_sum;
  ag.hh_precision_avg = (prec_denom == 0.0) ? 1.0 : (tp_sum / prec_denom);
  ag.hh_recall_avg    = (rec_denom  == 0.0) ? 1.0 : (tp_sum / rec_denom);
  const double f1_denom = ag.hh_precision_avg + ag.hh_recall_avg;
  ag.hh_f1_avg = (f1_denom > 0.0) ? (2.0 * ag.hh_precision_avg * ag.hh_recall_avg / f1_denom) : 0.0;

  ag.aae_avg = (hh_items == 0) ? 0.0 : (abs_err_sum / static_cast<double>(hh_items));
  ag.are_avg = (hh_items == 0) ? 0.0 : (rel_err_sum / static_cast<double>(hh_items));

  if (topk_total > 0) ag.topk_overlap_avg = topk_weighted / static_cast<double>(topk_total);

  return ag;
}

} // namespace hh
