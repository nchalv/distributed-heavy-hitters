#pragma once
#include "hh/core/id128.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>

namespace hh {

class ArenaMap {
public:
  struct Ref { std::uint32_t off; std::uint32_t len; };

  Ref intern(const std::string& bytes) {
    Ref r{static_cast<std::uint32_t>(arena_.size()), static_cast<std::uint32_t>(bytes.size())};
    arena_.insert(arena_.end(), bytes.begin(), bytes.end());
    return r;
  }
  void bind(const Id128& id, Ref r) { id2ref_[id] = r; }
  void bind_bytes(const Id128& id, const std::string& bytes) {
    auto it = id2ref_.find(id);
    if (it != id2ref_.end()) {
      const Ref& old = it->second;
      if (old.len == bytes.size() &&
          old.off + old.len <= arena_.size() &&
          std::memcmp(arena_.data() + old.off, bytes.data(), old.len) == 0) {
        return; // already bound to identical key bytes
      }
    }
    bind(id, intern(bytes));
  }

  bool lookup(const Id128& id, std::string& out) const {
    auto it = id2ref_.find(id);
    if (it == id2ref_.end()) return false;
    const auto r = it->second;
    out.assign(arena_.data()+r.off, arena_.data()+r.off+r.len);
    return true;
  }

  void clear() { arena_.clear(); id2ref_.clear(); }
  std::size_t memory_bytes() const {
    std::size_t bytes = sizeof(*this);
    bytes += arena_.capacity() * sizeof(char);
    bytes += id2ref_.bucket_count() * sizeof(void*);
    bytes += id2ref_.size() * sizeof(decltype(id2ref_)::value_type);
    return bytes;
  }
private:
  std::vector<char> arena_;
  std::unordered_map<Id128,Ref,Id128Hash> id2ref_;
};

} // namespace hh
