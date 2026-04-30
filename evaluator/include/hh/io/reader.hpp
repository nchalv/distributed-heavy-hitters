#pragma once
#include <functional>
#include <string>
#include <cstddef>

namespace hh {

// on_event(win, partition, key, weight)
using EventFn = std::function<void(std::size_t, std::size_t, const std::string&, int)>;

struct JsonGzNestedReader {
  // Read either:
  // 1) a gzip-compressed JSON file with shape {win:{partition:[<key>|[key,weight],...]}}
  // 2) a directory of per-window files (e.g., window_000000.json.gz, window_000001.json.gz, ...)
  //
  // For directory mode, each file may be either nested (same shape as above) or a single-window
  // partition object {partition:[...]} in which case the window id is inferred from the filename
  // when possible (window_<num>.json.gz), else from lexical file order.
  //
  // Window and partition keys are read as strings; numeric keys are ordered numerically, others
  // fall back to lexical order and default to index 0. Returns the number of emitted events.
  static std::size_t read(const std::string& path, const EventFn& on_event);
};

} // namespace hh
