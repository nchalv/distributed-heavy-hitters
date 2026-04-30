#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>

namespace hh {

struct Id128 {
  std::array<std::uint8_t,16> b{};
  bool operator==(const Id128& o) const noexcept { return std::memcmp(b.data(), o.b.data(), 16)==0; }
};

struct Id128Hash {
  std::size_t operator()(const Id128& x) const noexcept {
    std::uint64_t a, c;
    std::memcpy(&a, x.b.data(), 8);
    std::memcpy(&c, x.b.data()+8, 8);
    a ^= (c<<1) | (c>>63);
    a ^= a>>33; a *= 0xff51afd7ed558ccdULL;
    a ^= a>>33; a *= 0xc4ceb9fe1a85ec53ULL;
    a ^= a>>33;
    return static_cast<std::size_t>(a);
  }
};

} // namespace hh

