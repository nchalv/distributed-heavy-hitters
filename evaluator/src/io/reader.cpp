#include "hh/io/reader.hpp"
#include <zstr.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <cctype>
#include <optional>   // <-- needed for std::optional
#include <algorithm>
#include <limits>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

inline std::optional<std::size_t> to_index(const std::string& s) {
  if (s.empty()) return std::nullopt;
  for (unsigned char c : s) if (!std::isdigit(c)) return std::nullopt;
  try { return static_cast<std::size_t>(std::stoull(s)); }
  catch (...) { return std::nullopt; }
}

struct KeyIdx {
  std::size_t idx;
  std::string key;
};

inline bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::optional<std::size_t> window_index_from_filename(const std::string& name) {
  // Expected form: window_000123.json.gz
  if (!ends_with(name, ".json.gz")) return std::nullopt;
  const std::string stem = name.substr(0, name.size() - std::string(".json.gz").size());
  constexpr const char* prefix = "window_";
  if (stem.rfind(prefix, 0) != 0) return std::nullopt;
  return to_index(stem.substr(std::char_traits<char>::length(prefix)));
}

std::vector<KeyIdx> sorted_object_keys(const json& obj) {
  std::vector<KeyIdx> keys;
  keys.reserve(obj.size());
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    const std::string k = it.key();
    if (auto oi = to_index(k)) keys.push_back({*oi, k});
    else keys.push_back({std::numeric_limits<std::size_t>::max(), k});
  }
  std::sort(keys.begin(), keys.end(), [](const KeyIdx& a, const KeyIdx& b){
    if (a.idx != b.idx) return a.idx < b.idx;
    return a.key < b.key;
  });
  return keys;
}

bool is_partition_object(const json& j) {
  if (!j.is_object()) return false;
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (!it.value().is_array()) return false;
  }
  return true;
}

std::size_t emit_partition_object(const json& jw, std::size_t win_num, const hh::EventFn& on_event) {
  auto part_keys = sorted_object_keys(jw);
  std::size_t emitted = 0;
  for (const auto& pk : part_keys) {
    const auto& jp = jw.at(pk.key);
    if (!jp.is_array()) continue;
    const std::size_t part_num = (pk.idx == std::numeric_limits<std::size_t>::max()) ? 0 : pk.idx;
    for (const auto& je : jp) {
      if (je.is_string()) {
        on_event(win_num, part_num, je.get<std::string>(), 1);
        ++emitted;
      } else if (je.is_array() && je.size() == 2 && je[0].is_string() && je[1].is_number_integer()) {
        on_event(win_num, part_num, je[0].get<std::string>(), je[1].get<int>());
        ++emitted;
      } else {
        // ignore malformed entries
      }
    }
  }
  return emitted;
}

std::size_t read_single_json_gz(const std::string& path,
                                const hh::EventFn& on_event,
                                const std::optional<std::size_t>& forced_window = std::nullopt)
{
  std::ifstream fin(path, std::ios::binary);
  if (!fin) throw std::runtime_error("cannot open: " + path);
  zstr::istream zin(fin);

  json j;
  zin >> j;

  if (!j.is_object()) {
    throw std::runtime_error("top-level JSON must be an object {window:{partition:[keys]}} or {partition:[keys]}");
  }

  // Per-window file shape: {partition:[...]}
  if (is_partition_object(j)) {
    const std::size_t win_num = forced_window.value_or(0);
    return emit_partition_object(j, win_num, on_event);
  }

  // Original nested shape: {window:{partition:[...]}}
  auto win_keys = sorted_object_keys(j);
  std::size_t emitted = 0;
  const bool override_single_window = forced_window.has_value() && win_keys.size() == 1;
  for (const auto& wk : win_keys) {
    const auto& jw = j.at(wk.key);
    if (!jw.is_object()) continue;
    const std::size_t parsed_win = (wk.idx == std::numeric_limits<std::size_t>::max()) ? 0 : wk.idx;
    const std::size_t win_num = override_single_window ? *forced_window : parsed_win;
    emitted += emit_partition_object(jw, win_num, on_event);
  }
  return emitted;
}

} // namespace

namespace hh {

std::size_t JsonGzNestedReader::read(const std::string& path, const EventFn& on_event)
{
  const fs::path p(path);
  std::error_code ec;
  if (fs::is_directory(p, ec)) {
    std::vector<fs::path> files;
    for (const auto& ent : fs::directory_iterator(p)) {
      if (!ent.is_regular_file()) continue;
      const auto name = ent.path().filename().string();
      if (!ends_with(name, ".json.gz")) continue;
      files.push_back(ent.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b){
      return a.filename().string() < b.filename().string();
    });
    if (files.empty()) throw std::runtime_error("no .json.gz files found in directory: " + path);

    std::size_t emitted = 0;
    for (std::size_t i = 0; i < files.size(); ++i) {
      const auto& fp = files[i];
      std::optional<std::size_t> forced_win = window_index_from_filename(fp.filename().string());
      if (!forced_win) forced_win = i;
      emitted += read_single_json_gz(fp.string(), on_event, forced_win);
    }
    return emitted;
  }

  return read_single_json_gz(path, on_event, std::nullopt);
}

} // namespace hh
