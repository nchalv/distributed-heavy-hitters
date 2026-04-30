#include "hh/core/hash.hpp"
#include <array>
#include <cstring>
extern "C" {
#include "blake3.h"
}


namespace {
std::array<std::uint8_t,32> gK_id{}, gK_fp{}, gK_idx{};
inline void blake3_keyed(uint8_t out[], size_t outlen,
                         const uint8_t key[32],
                         const void* p1, size_t n1,
                         const void* p2=nullptr, size_t n2=0) {
  blake3_hasher h;
  blake3_hasher_init_keyed(&h, key);
  if (p1 && n1) blake3_hasher_update(&h, p1, n1);
  if (p2 && n2) blake3_hasher_update(&h, p2, n2);
  blake3_hasher_finalize(&h, out, outlen);
}
}

namespace hh {

void set_secret_keys(std::array<std::uint8_t,32> K_id,
                     std::array<std::uint8_t,32> K_fp,
                     std::array<std::uint8_t,32> K_idx) {
  gK_id = K_id; gK_fp = K_fp; gK_idx = K_idx;
}

Id128 id128_for(std::string_view norm) {
  Id128 id;
  static const std::uint8_t ctx[] = "HH-ID-v1";
  blake3_keyed(id.b.data(), 16, gK_id.data(), ctx, sizeof(ctx)-1, norm.data(), norm.size());
  return id;
}

std::uint16_t fp16_from_id(const Id128& id) {
  static const std::uint8_t ctx[] = "HH-FP-v1";
  std::uint8_t out[2];
  blake3_keyed(out, 2, gK_fp.data(), ctx, sizeof(ctx)-1, id.b.data(), id.b.size());
  return static_cast<std::uint16_t>(out[0] | (out[1]<<8));
}

IdxPair indices_from_id(const Id128& id, std::uint16_t fp16, std::size_t B) {
  static const std::uint8_t ctx[] = "HH-IDX-v1";
  std::uint8_t out[8];
  blake3_keyed(out, 8, gK_idx.data(), ctx, sizeof(ctx)-1, id.b.data(), id.b.size());
  std::uint64_t h; std::memcpy(&h, out, 8);
  std::size_t mask = B - 1;
  std::size_t i1 = static_cast<std::size_t>((h >> 32) & mask);
  std::size_t i2 = (i1 ^ (static_cast<std::uint32_t>(0x5bd1e995u) * fp16)) & mask; // CHK-style alt index
  return {i1, i2};
}

} // namespace hh

